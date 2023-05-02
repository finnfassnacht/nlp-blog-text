# Deploying NLP models to IBM Code Engine
Recently, there has been an explosion of popularity around AI, particularly NLP Models, due to their ability to interact with text in a human-like way and perform various language tasks with remarkable accuracy. Huggingface is an open-source library for natural language processing that has gained widespread popularity due to its extensive collection of pre-trained models and user-friendly tools.

In this blog post, we'll provide a step-by-step guide on deploying NLP Models from Huggingface to IBM Code Engine. We'll also include code snippets to help you follow along and get started with deploying Huggingface models to IBM Code Engine.
For this tutorial, we'll be using Python3 as our programming language, and our NLP model of choice will be the LLM GPT-Neo-125M by EleutherAI.

## Prerequisites
* Basic Knowledge about Huggingface
* Basic Knowledge about container images
* Knowledge of IBM Code Engine and how to deploy a basic app (find out [here](https://www.ibm.com/cloud/blog/deploying-a-simple-http-server-to-ibm-cloud-code-engine-from-source-code-using-python-node-and-go))
* An IBM Cloud account with sufficient privileges to create and manage resources

## Get your Model

Before we can begin deploying or generating text using the GPT-Neo 125M model, we need to obtain the model files. These files can be downloaded using git from the EleutherAI [GitHub repository](https://huggingface.co/EleutherAI/gpt-neo-125m/tree/main) on Huggingface's website. Once downloaded, we can begin working with the model.
1. Make sure that you have git LFS (large file storage) installed
```
git lfs install
```
2. Git clone the Model
```
git clone https://huggingface.co/EleutherAI/gpt-neo-125m
```
3. Delete unnecessary files
```
cd gpt-neo-125m && rm README.md rust_model.ot flax_model.msgpack
```
Now that you have downloaded the GPT-Neo 125M model files, we can begin using the model to generate text on your local machine.

## How to Run Your Model

Before we can start using the our model to generate text, we need to ensure that we have all the necessary packages install.

To do that we need to install Pytorch and Transformers.
1. Install PyTorch for CPUs by running the following command:
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```
2. Install the Transformers library by running the following command:
```
pip3 install transformers
```
Once these packages are installed, create a new Python file in the same directory where you downloaded the GPT-Neo 125M model files. This file will be used to write the code for generating text using the model.

**Here's some sample code you can use:**
```python
# Import the pipeline module, which bundles all files in your model directory together
from transformers import pipeline 

# Specify the model directory and task
generator = pipeline(task='text-generation', model='model/gpt-neo-125m') 

# Generate text with the prompt "hello my name is"
res = generator('hello my name is', max_length=20, do_sample=True, temperature=0.9, pad_token_id=50256)

# Print the generated text
print(res)
```
4. Save & Execute, congratulations you just generated Text with your local Large Language Model !

## Make a server
To interact with the Large Language Model (LLM) post-deployment, we need to set up a server that can handle requests and return generated text from the model. This can be achieved by creating a simple HTTP server on port 8080 and serving a single route that accepts prompts and returns the  corresponding generated text. We will then combine the server with our previous text generation program to enable it to return generated text.

**Sample sever with previous code integrated:**
```python

from flask import Flask
from flask import request
import os
from transformers import pipeline 
# load the model 
generator = pipeline(task='text-generation', model='model/gpt-neo-125m')
# start the server after to model is fully loaded
app = Flask(__name__)
# set up a route at /api/gpt-neo/
@app.route('/api/gpt-neo/', methods=['GET'])
def foo():
    # get argumetns from user
    data = request.args.to_dict()
    # run text generation
    res = generator(data["prompt"], max_length=20, do_sample=True, temperature=0.9,  pad_token_id=50256) # generate
    # return text back to user
    return {"llm":(res[0]["generated_text"])}
    
# Get the PORT from environment
port = os.getenv('PORT', '8080')
if __name__ == "__main__":
	app.run(host='0.0.0.0',port=int(port))
```
After Saving and Executing you should be able to curl your route.
```
ccurl http://127.0.0.1:8080/api/gpt-neo/?prompt=hello
```
Or Open it in your Browser [here](http://127.0.0.1:8080/api/gpt-neo/?prompt=hello)

## Deployment Options
Now that we know how to install and use our model locally, let's talk about how we can run it on Code Engine. 

Code Engine enables you to take or create an image and run it on containers to deploy your app. Each container has a bit of storage called "ephemeral storage," which is like a little hard drive for each container. We can use this storage to store our model, however it's important to note that when a container is terminated, its ephemeral storage is also terminated.

### We have two main Options to deploy our model
**Including Your Model in the Container Image**

A method to deploy your model to the cloud is to include the model in the container image. Do this by  including your Model alongside your code when building the container image. This way, when the container is launched, the model is available immediately, and you can start generating text right away.

>Pros:

* If your model is included in the container image, you can start generating text immediately after the container is launched.

>Cons:

* Including models in the container image increases the image size, leading to longer launching times, and in some cases, the size of the model may prevent launching the container altogether.

**Including Your Compressed Model in the Container Image**

Another method to deploy your model to the cloud is to include a compressed version of the model in the container image. Do this by including a compressed version of your Model alongside your code when building the container image. When the container is launched, the model is decompressed and then used.

**Previous code with decompression integrated:**
```python

from flask import Flask
from flask import request
import os
from transformers import pipeline 
import tarfile
# load the model 
# open and extract the model
file = tarfile.open('gpt-neo-125m.tar.gz')
file.extractall('model')
file.close()
generator = pipeline(task='text-generation', model='model/gpt-neo-125m')
# start the server after to model is fully loaded
app = Flask(__name__)
# set up a route at /api/gpt-neo/
@app.route('/api/gpt-neo/', methods=['GET'])
def foo():
    # get argumetns from user
    data = request.args.to_dict()
    # run text generation
    res = generator(data["prompt"], max_length=20, do_sample=True, temperature=0.9,  pad_token_id=50256) # generate
    # return text back to user
    return {"llm":(res[0]["generated_text"])}
    
# Get the PORT from environment
port = os.getenv('PORT', '8080')
if __name__ == "__main__":
	app.run(host='0.0.0.0',port=int(port))

```



>Pros:

* Compressing your Model leads to a small image size and therefore faster launching.

>Cons:
* Model needs to be decompressed fist


It's important to choose the deployment option that best suits your specific use case. As a rule of thumb, if you have a relatively small model (100MB-200MB), option one (including the model in the container image) may be the best choice. If you have a larger-sized model (500MB-1.5GB), option two (including the compressed model in the container image) may be more suitable. Ultimately, the decision will depend on the specific details of your use case. Keep in mind that Code Engine only supports CPUs and no GPUs, which means that Code Engine isn't suited for really big models anyway.

## Creating your Image


I will be using Podman to build images. If you don't have it installed already, you can do so [here](https://podman.io/getting-started/installation). Alternatively, you can use Docker.

Pull a pre-built PyTorch base image to use as the foundation for our container. We can do this by running the following command:
```
podman pull docker.io/cnstark/pytorch:1.13.1-py3.9.12-ubuntu20.04
```
* Create a requirements.txt file in the root directory of your project with the following content:
```
Flask
transformers
```
* Create a Dockerfile in the root directory of your project (note that "Dockerfile" has no file extension) with the following content:
```Dockerfile
# use a prebuilt pytorch image
FROM cnstark/pytorch:1.13.1-py3.9.12-ubuntu20.04
# define the working dir
WORKDIR /app
# Copy and install requirements
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
# Copy your Code and Model
COPY . .
# Command to starts
CMD ["python3","main.py"]
```
* Build your Image
```
podman build . -t --namespace--/gpt-neo
```

> Replace --namespace-- with your IBM container registry namespace name.

* Tag your Image 
```
podman tag --namespace--/gpt-neo:latest icr.io/--namespace--/gpt-neo:v1
```
>Replace --namespace-- with your IBM container registry namespace name.

>Note that icr.io is the global IBM Cloud Container Registry, and the name of your Container Registry may be different.

Once you have completed these steps, you should have a container image with your PyTorch model and Flask server ready to be deployed.
## Deploy your Image

* ### Push your Image to the Container Registry
1. Login to the CLI
```
ibmcloud login
```
2. If you don't have it already install the Container Registry Plugin
```
ibmcloud plugin install container-registry
```
3. Authorize your Client
```
ibmcloud cr login --client podman
```
4. Push your Image to the Container Registry
```
podman push icr.io/--namespace--/gpttest:v1
```
## Create your application 

* ### Setup a Registry Secret
Whenever possible, it is best to use a private container registry to store your container images. A private container registry can only be accessed within the IBM Cloud, which means that all traffic is routed through the cloud and not the public internet. This results in faster speeds and better security. Using a private container registry requires a registry-secret.

**Using the Web UI**

To create a registry secret:
1. Navigate to "Projects" in the Web UI 
2. Click on your project.
3. Select "Registry access" from the Side bar and click on "Create".
4. Select the "Custom" option and specify a name i.e "myaccess".
5. Then specify your Container Registry i.e "private.icr.io" and hit Create.

**Using the CLI**
```
ibmcloud ce secret create --format registry --name myaccess --server private.icr.io --username iamapikey --password <yourpassword>
```
Now that we have a access to your private container registry we can launch our App.

**Using the CLI**
```
ibmcloud ce app create --name gptzip --image private.icr.io/mynamespace1/gptzip:v1 --registry-secret myaccess --es 2G
```
> ```--image``` defines the location of the image

> ```--registry-secret``` defines the registry-secret which is required, when using a private container registry

> ```--es``` defines the ephemeral storage, 2GB should be enough

**Using the Web UI**
1. Navigate to "Projects" in the Web UI 
2. Click on your project and then on "Applications".
3. Select the "Create" button, and pick a name for the App
4. Click on the "Configure image" button
5. Select your Registry server (i.e "private.icr.io") from the dropdown
6. Enter your Registry access secret or leave "Code Engine managed secret" as-is
7. Confirm your namespace and image name
8. Select "v1" as your Tag
9. Click "Done" and then "Runtime settings"
10. Finally select 2GB in Ephemeral storage and click "Create"

After the Code Engine is finished deploying your App, you can play around with your Model online!!
Congratulations you just hosted your NLP model using IBM Code Engine


## In conclusion
In this blog post, we explored how to deploy NLP models from Huggingface to IBM Cloud Code Engine. We provided sample code for using your Model locally and deploying it in a server configuration. We also talked about methods of deployment and possible limitations. With the steps outlined in this post, you should have a clear understanding of how to deploy your NLP models to Code Engine.
