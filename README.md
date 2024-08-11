# Medical RAG using Qdrant and Gradio
This project implemetns retireival augmented generation to create a medical assistant chatbot with the help of ```Llama3-OpenBioLLM-70B```. The embeddings are stored in Qdrant vector database.  To learn more about the project please refer this [article](link).

![Alt Text - description of the image](https://github.com/vansh-khaneja/Medical-RAG-using-Llama3-OpenBioLLM-70B/blob/main/image/img1.png?raw=true)


## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Execution](#execution)
- [Contact](#contact)

## Introduction

In this project, we used ```Llama3-OpenBioLLM-70B``` as the langauge model to create the chatbot and ```all-mpnet-base-v2``` for creating vector embedding to store in the database.

## Features

- Fast and efficient QnA answering
- Supports `Llama3-OpenBioLLM-70B` and other medical LLMs
- Accurate Data retrieval and generation
- User Interactive Frontend 

## Installation

1. Clone the repository:

    ```sh
    git clone https://github.com/vansh-khaneja/Medical-RAG-using-Qdrant
    cd Medical-RAG-using-Qdrant
    ```

2. Set up the Python environment and install dependencies:

    ```sh
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

3. Set up Qdrant:

    Follow the [Qdrant documentation](https://qdrant.tech/documentation/) to install and configure Qdrant on your system.

## Execution
1.Download and load the dataset for this project [here](https://www.kaggle.com/datasets/thedevastator/comprehensive-medical-q-a-dataset) or you can try with your own dataset.

```sh
    df = pd.read_csv('/train.csv')
```


2.Execute the ```main.py``` file by running this command in terminal.

```sh
    python main.py
```


## Contact

For any questions or issues, feel free to open an issue on this repository or contact me at vanshkhaneja2004@gmail.com.

Happy coding!
