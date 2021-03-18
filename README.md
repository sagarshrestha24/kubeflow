# End-to-end Mnist Pipeline
## Introduction
With Kubeflow Pipelines you can build entire workflows that automate the steps involved in going from training a machine learning model to actually serving an optimized version of it. These steps can be triggered automatically by a CI/CD workflow or on demand from a command line or notebook.
Kubeflow Pipelines (kfp) comes with a user interface for managing and tracking experiments, jobs, and runs. A pipeline is a description of a machine learning workflow, replete with all inputs and outputs. In Kubeflow Pipelines, an experiment is a workspace where you can experiment with different configuration of your pipelines. Experiments are a way to organize runs of jobs into logical groups. A run is simply a single execution (instance) of a pipeline. Kubeflow Pipelines also supports recurring runs, which is a repeatable run of a pipeline. Based on a so-called run trigger an instance of a pipeline with its run configuration is periodically started. As of now, run triggers are time-based (i.e. not event-based).
This notebook trains a simple (MNIST) model in TensorFlow and serves it with KFServing, which is a serverless inference server. What this means is that you do not have to worry about which machines it runs on, networking, autoscaling, health checks, and what have you. Instead, you can focus on what matters to you: the model and a REST API you can call for predictions. If you are familiar with Kubernetes, you can even do out-of-the-box canary deployments, in which a percentage of traffic is directed to the 'canary (in the coal mine)' with the latest model to ensure it functions properly before completely rolling out any (potentially problematic) updates.
If you prefer to use a more sophisticated model or a PyTorch-based one, you can check out the relevant notebooks: MNIST with TensorFlow or MNIST with PyTorch.
KFServing reads the model file from MinIO, an open-source S3-compliant object storage tool, which is already included with your Kubeflow installation.
We also use MinIO to provide the input data set to the pipeline. This way it can run without a connection to the Internet.
Let's make sure Kubeflow Pipelines is available:



## Copy input data set into MinIO using its CLI
First, we configure credentials for mc, the MinIO command line client. We then use it to create a bucket, upload the dataset to it, and set access policy so that the pipeline can download it from MinIO.

          ./mc alias set minio http://minio-service.kubeflow:9000 mlopstestnplink D6N1ErV++1pUSrjGPZCW48UKMSEzxMf1884l5j/eqk99ZkIMbgpmUTRFs3zPsZGWX42iD7IwdwDTxr9ZNHPTeA==


We have to create a bucket:

    ./mc mb minio/database
    tar --dereference -czf datasets.tar.gz ./datasets
     ./mc cp datasets.tar.gz minio/database/datasets.tar.gz
    ./mc policy set download minio/database

## How to Implement Kubeflow Pipelines Components

As we said before, components are self-contained pieces of code: Python functions.
The function must be completely self-contained. No code (incl. imports) can be defined outside of the body itself. All imports must be included in the function body itself! Imported packages must be available in the base image.

Why? Because each component will be packaged as a Docker image. The base image must therefore contain all dependencies. Any dependencies you install manually in the notebook are invisible to the Python function once it's inside the image. The function itself becomes the entrypoint of the image, which is why all auxiliary functions must be defined inside the function. That does cause some unfortunate duplication, but it also means you do not have to worry about the mechanism of packaging, as we shall see below.

For our pipeline, we shall define four components:

    Download the MNIST data set
    Train the TensorFlow model
    Evaluate the trained model
    Export the trained model
    Serve the trained model

We also need the current Kubernetes namespace, which we can dynamically grab using Kubeflow Fairing.

    from typing import NamedTuple
    import kfp
    import kfp.components as components
    import kfp.dsl as dsl
    import kubeflow.fairing.utils

    from kfp.components import InputPath, OutputPath

    NAMESPACE = kubeflow.fairing.utils.get_current_k8s_namespace()
    
 Function arguments specified with InputPath and OutputPath are the key to defining dependencies. For now, it suffices to think of them as the input and output of each step. 
 
 ## Component 1: Download the MNIST Data Set
 
     def download_dataset(data_dir: OutputPath(str)):
        """Download the MNIST data set to the KFP volume to share it among all steps"""
        import urllib.request
        import tarfile
        import os

        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        url = "http://minio-service.kubeflow:9000/database/datasets.tar.gz"
        stream = urllib.request.urlopen(url)
        tar = tarfile.open(fileobj=stream, mode="r|gz")
        tar.extractall(path=data_dir)
        
 ## Component 2: Train the Model
 
 For both the training and evaluation we must divide the integer-valued pixel values by 255 to scale all values into the [0, 1] (floating-point) range. This function must be copied into both component functions (cf. normalize_image).

If you wish to learn more about the model code, please have a look at the MNIST with TensorFlow notebook.

    def train_model(data_dir: InputPath(str), model_dir: OutputPath(str)):
        """Trains a single-layer CNN for 5 epochs using a pre-downloaded dataset.
        Once trained, the model is persisted to `model_dir`."""

        import os
        import tensorflow as tf
        import tensorflow_datasets as tfds

        def normalize_image(image, label):
            """Normalizes images: `uint8` -> `float32`"""
            return tf.cast(image, tf.float32) / 255.0, label

        model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dense(10, activation="softmax"),
            ]
        )
        model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer=tf.keras.optimizers.Adam(0.001),
            metrics=["accuracy"],
        )

        print(model.summary())
        ds_train, ds_info = tfds.load(
            "mnist",
            split="train",
            shuffle_files=True,
            as_supervised=True,
            with_info=True,
            download=False,
            data_dir=f"{data_dir}/datasets",
        )

        # See: https://www.tensorflow.org/datasets/keras_example#build_training_pipeline
        ds_train = ds_train.map(
            normalize_image, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        ds_train = ds_train.cache()
        ds_train = ds_train.shuffle(ds_info.splits["train"].num_examples)
        ds_train = ds_train.batch(128)
        ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

        model.fit(
            ds_train,
            epochs=5,
        )

        model.save(model_dir)
        print(f"Model saved {model_dir}")
        print(os.listdir(model_dir))


## Component 3: Evaluate the Model


      def evaluate_model(
          data_dir: InputPath(str), model_dir: InputPath(str), metrics_path: OutputPath(str)
      ) -> NamedTuple("EvaluationOutput", [("mlpipeline_metrics", "Metrics")]):
          """Loads a saved model from file and uses a pre-downloaded dataset for evaluation.
          Model metrics are persisted to `/mlpipeline-metrics.json` for Kubeflow Pipelines
          metadata."""

      import json
      import tensorflow as tf
      import tensorflow_datasets as tfds
      from collections import namedtuple

      def normalize_image(image, label):
          return tf.cast(image, tf.float32) / 255.0, label

      ds_test, ds_info = tfds.load(
          "mnist",
          split="test",
          shuffle_files=True,
          as_supervised=True,
          with_info=True,
          download=False,
          data_dir=f"{data_dir}/datasets",
      )

      # See: https://www.tensorflow.org/datasets/keras_example#build_training_pipeline
      ds_test = ds_test.map(
          normalize_image, num_parallel_calls=tf.data.experimental.AUTOTUNE
      )
      ds_test = ds_test.batch(128)
      ds_test = ds_test.cache()
      ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

      model = tf.keras.models.load_model(model_dir)
      (loss, accuracy) = model.evaluate(ds_test)

      metrics = {
          "metrics": [
              {"name": "loss", "numberValue": str(loss), "format": "PERCENTAGE"},
              {"name": "accuracy", "numberValue": str(accuracy), "format": "PERCENTAGE"},
          ]
      }

      with open(metrics_path, "w") as f:
          json.dump(metrics, f)

      out_tuple = namedtuple("EvaluationOutput", ["mlpipeline_metrics"])

      return out_tuple(json.dumps(metrics))
      
## Hyperparameter Tuning with Katib
## Introduction

Hyperparameter tuning is the process of optimizing a model's hyperparameter values in order to maximize the predictive quality of the model. Examples of such hyperparameters are the learning rate, neural architecture depth (layers) and width (nodes), epochs, batch size, dropout rate, and activation functions. These are the parameters that are set prior to training; unlike the model parameters (weights and biases), these do not change during the process of training the model.

Katib automates the process of hyperparameter tuning by running a pre-configured number of training jobs (known as trials) in parallel. Each trial evaluates a different set of hyperparameter configurations. Within each experiment it automatically adjusts the hyperparameters to find their optimal values with regard to the objective function, which is typically the model's metric (e.g. accuracy, AUC, F1, precision). An experiment therefore consists of an objective, a search space for the hyperparameters, and a search algorithm. At the end of the experiment, Katib outputs the optimized values, which are also known as suggestions.
Three Data Sets
Whereas it is common to have training and test data sets in traditional (supervised) machine learning, in deep learning (esp. when combined with hyperparameter tuning), it is recommended to have a three-way split: training, validation (a.k.a. as development), and test. The training data set is, as always, to learn parameters (weights and biases) from data. The test data set is also known as the hold-out set and its sole purpose is to check the model's hypothesis of parameter values in terms of how well it generalizes to data it has never come across. The point of the validation data set is to cross-validate the model and tweak the hyperparameters. Since information from this data set is used to adjust the model, it is not an objective test of the model's generalizability. It is not unlike a teacher checking up on students:

    

What You'll Learn

This notebook shows how you can create and configure an Experiment for both TensorFlow and PyTorch training jobs. In terms of Kubernetes, such an experiment is a custom resource handled by the Katib operator.
What You'll Need

A Docker image with either a TensorFlow or PyTorch model that accepts hyperparameters as arguments. Please click on the links to see such models.

That's it, so let's get started!
How to Specify Hyperparameters in Your Models

In order for Katib to be able to tweak hyperparameters it needs to know what these are called in the model. Beyond that, the model must specify these hyperparameters either as regular (command line) parameters or as environment variables. Since the model needs to be containerized, any command line parameters or environment variables must to be passed to the container that holds your model. By far the most common and also the recommended way is to use command line parameters that are captured with argparse or similar; the trainer (function) then uses their values internally.
How to Expose Model Metrics as Objective Functions

By default, Katib collects metrics from the standard output of a job container by using a sidecar container. In order to make the metrics available to Katib, they must be logged to stdout in the key=value format. The job output will be redirected to /var/log/katib/metrics.log file. This means that the objective function (for Katib) must match the metric's key in the models output. It's therefore possible to define custom model metrics for your use case.
Sidecars
In the sidecar (a.k.a. sidekick or decomposition) pattern, if you are not already familiar with it, a secondary (sidecar) container is attached to the primary workload inside a pod in Kubernetes. In many cases, pods run a single container, but peripheral services, such as networking services, monitoring, and logging, are required in all applications and services. With sidecars there is no need to re-implement basic but secondary tasks in each service or application. The sidecar has the same lifecycle as the primary application and it has access to the same resources. The sidecar is, however, isolated from the main container, which means it does not have to be implemented in the same technology. This means it can easily be reused across various workloads.

Katib does not care whether you use TensorFlow, PyTorch, MXNet, or any other framework for that matter. All it needs to do its job is a (parameterized) trainer container and the logs to grab the model's metrics from.
How to Create Experiments¶

Before we proceed, let's set up a few basic definitions that we can re-use. Note that you typically use (YAML) resource definitions for Kubernetes from the command line, but we shall show you how to do everything from a notebook.

    kubectl apply -f katib.yaml


## Component 4: Export the Model

    def export_model(
        model_dir: InputPath(str),
        metrics: InputPath(str),
        export_bucket: str,
        model_name: str,
        model_version: int,
    ):
        import os
        import boto3
        from botocore.client import Config

        s3 = boto3.client(
            "s3",
            endpoint_url="http://minio-service.kubeflow:9000",
            aws_access_key_id="mlopstestnplink",
            aws_secret_access_key="D6N1ErV++1pUSrjGPZCW48UKMSEzxMf1884l5j/eqk99ZkIMbgpmUTRFs3zPsZGWX42iD7IwdwDTxr9ZNHPTeA==",
            config=Config(signature_version="s3v4"),
        )

        # Create export bucket if it does not yet exist
        response = s3.list_buckets()
        export_bucket_exists = False

        for bucket in response["Buckets"]:
            if bucket["Name"] == export_bucket:
                export_bucket_exists = True

        if not export_bucket_exists:
            s3.create_bucket(ACL="public-read-write", Bucket=export_bucket)

        # Save model files to S3
        for root, dirs, files in os.walk(model_dir):
            for filename in files:
                local_path = os.path.join(root, filename)
                s3_path = os.path.relpath(local_path, model_dir)

                s3.upload_file(
                    local_path,
                    export_bucket,
                    f"{model_name}/{model_version}/{s3_path}",
                    ExtraArgs={"ACL": "public-read"},
                )

        response = s3.list_objects(Bucket=export_bucket)
        print(f"All objects in {export_bucket}:")
        for file in response["Contents"]:
            print("{}/{}".format(export_bucket, file["Key"]))
            
            
## Metadata SDK
## Introduction

All information about executions, models, data sets as well as the files and objects that are a part of a machine learning workflow are referred to as metadata. The Metadata SDK allows you to manage all ML assets:

    An Execution captures metadata of a single run of an ML workflow, which can be either a pipeline or a notebook. Any derived data that is used or produced in the context of a single execution is referred to as an artifact.
    Metadata of a Model includes a URI to its location, a name and description, training framework (e.g. TensorFlow, PyTorch, MXNet), hyperparameters and their values, and so on.
    Metrics collect evaluation metrics of the model
    A DataSet describes the data that is either the input or output of a component within an ML workflow.

Behind the scenes, the Metadata SDK uses the gRPC service of MLMD, the ML Metadata library, which was originally designed for TFX (TensorFlow eXtended) and offers both implementations for SQLite and MySQL.

With the Metadata SDK you can also add so-called metadata watchers to check up on Kubernetes resource changes and to save the related data in the metadata service.
What You'll Learn

In this notebook, you'll learn how to use the Metadata SDK to display information about executions and interact with the metadata available within Kubeflow.
What You'll Need

Nothing except this notebook.
How to Create a Workspace

A workspace is a grouping of pipelines, notebooks, and their artifacts. A single workspace can hold multiple executions.

To define various objects (e.g. executions, runs, models) you therefore need to create a workspace. Unless you define multiple workspaces within the same context, you do not have to specify it after you have created

Let's import the metadata modules and store the default DNS for the host as well as the port for the metadata store in a couple of variables


    from kubeflow.metadata import metadata

    METADATA_STORE_HOST = "metadata-grpc-service.kubeflow"

    METADATA_STORE_PORT = 8080

    METADATA_STORE = metadata.Store(

        grpc_host=METADATA_STORE_HOST, grpc_port=METADATA_STORE_PORT)

    ws1 = metadata.Workspace(
        # Connect to metadata service in namespace kubeflow in k8s cluster.
        store=metadata.Store(grpc_host=METADATA_STORE_HOST, grpc_port=METADATA_STORE_PORT),
        name="Mnist Workspace",
    description="Artifact on Mnist Workspace",
    labels={"n1": "v1"})
    

## How to Create a Run in a Workspace

The difference between runs and executions is subtle: an execution records the run of a component or step in a machine learning workflow (along with its runtime parameters).

A run is an instance of an executable step.

An execution therefore always refers to a run.

We'll also define a helper function:

    from uuid import uuid4

    def add_suffix(name: str) -> str:

        """

        Appends an underscore and hexidecimal UUID to `name`

        :param str name: String to be suffixed

        :return: Suffixed string

        :rtype: str

        """

        return f"{name}_{uuid4().hex}"


    run = metadata.Run(

        workspace=ws1,

        name=add_suffix("run"),

        description="A run in our workspace",
        )


## How to Create an Execution of a Run

    exec = metadata.Execution(

        name=add_suffix("execution"),

        workspace=ws1,

        run=run,

        description="An execution of our run",

    )


    print(f"Execution ID: {exec.id}")

## How to Log Artifacts for an Execution

An execution can have both input and output artifacts. Artifacts that can be logged for executions are Model, DataSet, Metrics, or a custom artifact type.

You can see defined artifacts by navigating to the Kubeflow Central Dashboard's Artifact Store.
How to Log a Data Set

A data set that is used by the model itself is an input artifact. It can be registered as follows:

    date_set_version = add_suffix("ds")

    data_set = exec.log_input(

        metadata.DataSet(

            description="Sample data",

            name="mnist-example",

            owner="mnist@kubeflow.com",

            uri="file://path/to/dataset",

            version=date_set_version,

            query="SELECT * FROM mnist",

        )

    )



    print(f"Data set ID:      {data_set.id}")

    print(f"Data set version: {data_set.version}")


The data itself is available at the specified uri. The query is optional and documents how this data is fetched from the source. It is not used to retrieve it. After all, the data does not have to live in a relational database at all.
How to Log a Model

If a step of a machine learning workflow generates a model, it is logged as an output artifact:

    model_version = add_suffix("model")

    model = exec.log_output(

        metadata.Model(

            name="MNIST",

            description="Model to recognize handwritten digits",

            owner="owner@my-company.com",

            uri="gcs://my-bucket/mnist",

            model_type="neural network",

            training_framework={"name": "tensorflow", "version": "v1.0"},

            hyperparameters={

                "learning_rate": 0.5,

                "layers": [10, 3, 1],

                "early_stop": True,

            },

            version=model_version,

            labels={"a_label": "some-value"},

        )

    )

    

    print(f"Model ID:      {model.id}")

    print(f"Model version: {model.version}")

### How to Log the Evaluation of a Model

        metrics = exec.log_output(

            metadata.Metrics(

                name="MNIST evaluation",

                description="Evaluation of the MNIST model",

                owner="mnist@kubeflow.com",

                uri="s3://mnist/mnist",

                data_set_id=str(data_set.id),

                model_id=str(model.id),

                metrics_type=metadata.Metrics.VALIDATION,

                values={"accuracy": 0.95},

                labels={"mylabel": "l1"},

            )

            )



    print(f"Metrics ID: {metrics.id}")

Possible values for metrics_type:

    TRAINING
    VALIDATION
    TESTING
    PRODUCTION

If you are not familiar with the distinction between validation and training, please check out the notebook on hyperparameter tuning, which explains the difference and the need for an additional evaluation step.

### How to Add Metadata for Serving the Model

Once you're satisfied with the model, you want to serve it. The model server is an execution with a model as input artifact:

    app = metadata.Execution(

        name="Serving the MNIST model",

        workspace=ws1,

        description="An execution to represent the model serving component",

    )


    served_model = metadata.Model(

        name="MNIST",

        uri="s3://mnist/mnist",

        version=model.version,

    )


    m = app.log_input(served_model)


    print(f"Serving model with ID:      {m.id}")

    print(f"Serving model with version: {m.version}")

Please note that we use the name, uri, and version to identify the model. As stated before, only the first two are required, but it's a good practice to also include the version.

### How to List All Models in a Workspace

The Artifact Store is user interface that displays artifacts across all workspaces. Not all fields are available, which means we cannot filter easily on, say, custom labels.

Fortunately, we can ask for all artifacts of a certain type: Model, Metrics, DataSet, or a custom artifact. Here's how to list all models:

    artifacts = ws1.list(metadata.Model.ARTIFACT_TYPE_NAME)

    artifacts

    import pandas as pd

    pd.DataFrame.from_dict(artifacts)

You can see the output includes the labels. Labels are particularly helpful when monitoring many (versions of) models in production, both with regard to system and model performance, as both can affect the overall user experience; a bad prediction (e.g. recommendation) from a responsive service negatively affects the user experience, as does an unresponsive service with good predictions. Model as well as system performance metrics need to be tracked over time and across versions to ensure a solid user experience. With (shared) labels it's possible to monitor both simultaneously.
How to Track Lineage

The same is true of executions and artifacts that belong to certain models

    model_events = ws1.store.get_events_by_artifact_ids([model.id])


    execution_ids = set(e.execution_id for e in model_events)

    print(f"Executions related to the model: {execution_ids}")


    trainer_events = ws1.store.get_events_by_execution_ids([exec.id])

    artifact_ids = set(e.artifact_id for e in trainer_events)

    print(f"Artifacts related to the trainer: {artifact_ids}")

### Component 5: Serve the Model


Kubeflow Pipelines comes with [a pre-defined KFServing component](https://raw.githubusercontent.com/kubeflow/pipelines/f21e0fe726f8aec86165beca061f64fa730e0ac7/components/kubeflow/kfserving/component.yaml) which can be imported from GitHub repo and reused across the pipelines without

the need to define 

it every time. We include a copy with the tutorial to make it work in an air-gapped environment.

Here's what the import looks like:

kfserving = components.load_component_from_file("kfserving-component.yaml")

How to Combine the Components into a Pipeline

Note that up to

this point we have not yet used the Kubeflow Pipelines SDK!



With our four components (i.e. self-contained functions) defined, we can wire up the dependencies with Kubeflow Pipelines.


The call [`components.func_to_container_op(f, base_image=img)(*args)`](https://www.kubeflow.org/docs/pipelines/sdk/sdk-overview/) has the following ingredients:

- `f` is t

he Python function that defines a component

- `img` is the base (Docker) image used to package the function

- `*args` lists the arguments to `f`


What the `*args` mean is best explained by going forward through the graph:

- `downloadOp` is the very first step and has no dependencies; it therefore has no `InputPath`.

  Its output (i.e. `OutputPath`) is stored in `data_dir`.

- `trainOp` needs the data downloaded from `downloadOp` and its signature lists `data_dir` (input) and `model_dir` (output).

  So, it _depends on_ `downloadOp.output` (i.e. the previous step's output) and stores its own outputs in `model_dir`, which can be used by another step.

  `downloadOp` is the parent of `trainOp`, as required.

- `evaluateOp`'s function takes three arguments: `data_dir` (i.e. `downloadOp.output`), `model_dir` (i.e. `trainOp.output`), and `metrics_path`, which is where the function stores its evaluation metrics.

  That way, `evaluateOp` can only run after the successful completion of both `downloadOp` and `trainOp`.

- `exportOp` runs the function `export_model`, which accepts five parameters: `model_dir`, `metrics`, `export_bucket`, `model_name`, and `model_version`.

  From where do we get the `model_dir`?

  It is nothing but `trainOp.output`.

  Similarly, `metrics` is `evaluateOp.output`.

  The remaining three arguments are regular Python arguments that are static for the pipeline: they do not depend on any step's output being available.

  Hence, they are defined without using `InputPath`.
  

- `kfservingOp` is loaded from the external component and its order of execution should be specified explicitly by using `kfservingOp.after(evaluateOp)` function which assigns `exportOp` as a parent.


      def train_and_serve(

          data_dir: str,

          model_dir: str,

          export_bucket: str,

          model_name: str,

          model_version: int,

      ):

          # For GPU support, please add the "-gpu" suffix to the base image

          BASE_IMAGE = "mesosphere/kubeflow:1.0.1-0.5.0-tensorflow-2.2.0"



          downloadOp = components.func_to_container_op(

              download_dataset, base_image=BASE_IMAGE

          )()



          trainOp = components.func_to_container_op(train_model, base_image=BASE_IMAGE)(

              downloadOp.output

          )



          evaluateOp = components.func_to_container_op(evaluate_model, base_image=BASE_IMAGE)(

              downloadOp.output, trainOp.output

          )



          exportOp = components.func_to_container_op(export_model, base_image=BASE_IMAGE)(

              trainOp.output, evaluateOp.output, export_bucket, model_name, model_version

          )

      

          kfservingOp = kfserving(

              action="apply",

              model_name="mnist",

              default_model_uri=f"s3://{export_bucket}/{model_name}",

              canary_model_traffic_percentage="10",

              namespace="kubeflow",

              framework="sklearn",

              default_custom_model_spec="{}",

              canary_custom_model_spec="{}",

              autoscaling_target="0",

              kfserving_endpoint="",

          )


          kfservingOp.after(exportOp)

      def op_transformer(op):

          op.add_pod_annotation(name="sidecar.istio.io/inject", value="false")

          return op


      @dsl.pipeline(

          name="End-to-End MNIST Pipeline",

          description="A sample pipeline to demonstrate multi-step model training, evaluation, export, and serving",

      )

      def mnist_pipeline(

          model_dir: str = "/train/model",

          data_dir: str = "/train/data",

          export_bucket: str = "mnist",

          model_name: str = "mnist",

          model_version: int = 1,

      ):

          train_and_serve(

              data_dir=data_dir,

              model_dir=model_dir,

              export_bucket=export_bucket,

              model_name=model_name,

              model_version=model_version,

          )

          dsl.get_pipeline_conf().add_op_transformer(op_transformer)
          

With that in place, let's submit the pipeline directly from our notebook:

    pipeline_func = mnist_pipeline

    run_name = pipeline_func.__name__ + " run"

    experiment_name = "End-to-End MNIST Pipeline"

    arguments = {

        "model_dir": "/train/model",

        "data_dir": "/train/data",

        "export_bucket": "mnist",

        "model_name": "mnist",

        "model_version": "1",

    }



    client = kfp.Client()

    run_result = client.create_run_from_pipeline_func(

        pipeline_func,

        experiment_name=experiment_name,

        run_name=run_name,

        arguments=arguments,

    )
