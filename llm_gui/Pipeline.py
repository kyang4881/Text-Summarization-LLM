import time
import random
import re
import torch
import os
import json
from torch import cuda, bfloat16
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig
import bitsandbytes
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import shutil
from typing import Dict, List, Optional, Union, Literal, Tuple
import gradio as gr

class importData:
    """A class for compiling dataset from a specified file.

    Args:
        file_path (str): The path to the data file.
    """
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path

    def get_file(self) -> pd.DataFrame:
        """
        Reads the data from the file, processes it, and returns a DataFrame.

        The method reads a CSV file, processes the 'message' column by
        stripping leading and trailing spaces and prefixing it with 'Conversation: ',
        groups the data by 'conversation_id', and aggregates messages for each conversation.

        Returns:
            pd.DataFrame: A DataFrame with 'conversation_id' and aggregated 'message'.
        """
        data = pd.read_csv(self.file_path, sep=';', encoding='ISO-8859-1')
        data = pd.DataFrame(data)
        data['message'] = data['message'].astype(str)
        # Prefix 'message' with 'Conversation: ' and strip extra spaces
        data['message'] = data['message'].apply(lambda x: f"Conversation: {x.strip()}")
        # Group by 'conversation_id' and aggregate messages
        data = data.groupby('conversation_id').agg({'message': ' '.join}).reset_index()
        return data

class load_model:
    """A class for loading a model from Hugging Face.

    Args:
        hf_auth (str): The authentication token for Hugging Face.
        model_id (str): The identifier of the model to load (default is 'meta-llama/Llama-2-7b-chat-hf').
        quantization (bool): Whether to use quantization for the model (default is True).
        device (str): The device to load the model on (default is 'cuda').
    """
    def __init__(self,
                 hf_auth: str,
                 model_id: str = 'meta-llama/Llama-2-7b-chat-hf',
                 quantization: bool = True,
                 device: str = "cuda") -> None:
        self.model_id = model_id
        self.quantization = quantization
        self.device = device
        self.hf_auth = hf_auth

    def load(self) -> Tuple:
        """
        Loads the model and tokenizer from Hugging Face.

        If quantization is enabled, configures and loads a quantized version of the model.
        Otherwise, loads the model without quantization.

        Returns:
            Tuple[AutoModelForCausalLM, AutoTokenizer]: A tuple containing the loaded model and tokenizer.
        """
      
        # Load the model with or without quantization
        if self.quantization:
            #tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_auth_token=self.hf_auth)
            tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            qconfig = transformers.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=bfloat16
            )

            model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_id,
                quantization_config=qconfig
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            model = AutoModelForSeq2SeqLM.from_pretrained(self.model_id)

        model.eval() # Set the model to evaluation mode

        return model, tokenizer


class inference:
    """A class for making inferences using a given model.

    Args:
        data (pd.DataFrame): The dataset to make predictions on.
        tokenizer (AutoTokenizer): The tokenizer used for encoding input.
        input_col (str): The column name in the dataset that contains input text (default is "message").
        few_shot_examples (Dict): Examples for few-shot learning (default is an empty dictionary).
        truncation (bool): Whether to truncate input text (default is False).
        device (str): The device to run the model on (default is 'cuda').
        prompt_instruction (str): Instruction to prepend to each input text (default is a summarization prompt).
        model (str): The identifier of the model to use (default is 'meta-llama/Llama-2-7b-chat-hf').
        padding (bool): Whether to pad input text (default is False).
        max_length (int): The maximum length of the generated output (default is 128).
        max_new_tokens (int): The maximum number of new tokens to generate (default is 256).
    """
    def __init__(self,
                  data: pd.DataFrame,
                  tokenizer: AutoTokenizer,
                  input_col: str = "message",
                  few_shot_examples: Dict = {},
                  truncation: bool = False,
                  device: str = "cuda",
                  prompt_instruction: str = "The following text contains conversational data. Summarize the text in 2-3 sentences.",
                  model: str = 'meta-llama/Llama-2-7b-chat-hf',
                  padding: bool = False,
                  max_length: int = 128,
                  max_new_tokens: int = 256
    ) -> None:
        self.model = model
        self.data = data
        self.truncation = truncation
        self.padding = padding
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens
        self.tokenizer = tokenizer
        self.device = device
        self.input_col = input_col
        self.prompt_instruction = prompt_instruction
        self.few_shot_examples = few_shot_examples

    def generate_results(self) -> pd.DataFrame:
        """
        Generates inferences from the model for each row in the dataset.

        For each input text, creates a prompt, encodes the input, generates the output using the model,
        decodes the output, and appends it to the predictions list.

        Returns:
            pd.DataFrame: The original dataset with an additional column 'preds' containing the model's predictions.
        """
        start_time = time.time()
        preds = []
        for i in range(len(self.data)):
            prompt = f"""{self.prompt_instruction}/n/nText: {self.data[self.input_col][i]}/n\n"""
            encoded_inputs = self.tokenizer(prompt, truncation=self.truncation, max_length=self.max_length, padding=self.padding, return_tensors="pt").to(self.device)
            input_ids = encoded_inputs.input_ids.to(self.device)
            encoded_outputs = self.model.generate(input_ids, max_new_tokens=self.max_new_tokens)
            #decoded_outputs = self.tokenizer.decode(encoded_outputs[0][input_ids.shape[1] :], max_length=self.max_length, skip_special_tokens=True)
            decoded_outputs = self.tokenizer.decode(encoded_outputs[0], skip_special_tokens=True)
            preds.append(decoded_outputs)
            #print(prompt)

        self.data.loc[:, 'preds'] = preds
        end_time = time.time()
        print(f"\nTotal time taken: {(end_time-start_time)/60:.2f} minutes")

        return self.data

class pipeline:
    """A class for running an inference pipeline with data, model, and tokenizer.

    Args:
        n_inference_samples (int): The number of samples to use for inference.
        file_path (str): The path to the data file.
        model_id (str): The identifier of the model to use.
        quantization (bool): Whether to use quantization for the model.
        device (str): The device to run the model on.
        hf_auth (str): The authentication token for Hugging Face.
        truncation (bool): Whether to truncate input text.
        padding (bool): Whether to pad input text.
        max_length (int): The maximum length of the generated output.
        max_new_tokens (int): The maximum number of new tokens to generate.
        input_col (str): The column name in the dataset that contains input text.
        prompt_instruction (str): Instruction to prepend to each input text.
        few_shot_examples (Dict): Examples for few-shot learning.
    """
    def __init__(self,
                 n_inference_samples: int,
                 file_path: str,
                 model_id: str,
                 quantization: bool,
                 device: str,
                 hf_auth: str,
                 truncation: bool,
                 padding: bool,
                 max_length: int,
                 max_new_tokens: int,
                 input_col: str,
                 prompt_instruction: str,
                 few_shot_examples: Dict[str, None]) -> None:
        self.n_inference_samples = n_inference_samples
        self.file_path = file_path
        self.model_id = model_id
        self.quantization = quantization
        self.device = device
        self.hf_auth = hf_auth
        self.truncation = truncation
        self.padding = padding
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens
        self.input_col = input_col
        self.prompt_instruction = prompt_instruction
        self.few_shot_examples = few_shot_examples


    def run(self) -> pd.DataFrame:
        """
        Executes the inference pipeline.

        This method performs the following steps:
          1. Loads data from the specified file path.
          2. Loads the model and tokenizer.
          3. Performs inference on the data using the model and tokenizer.
          4. Returns the results with predictions.

        Returns:
            pd.DataFrame: The dataset with an additional column 'preds' containing the model's predictions.
        """
        # Load and preprocess data
        data = importData(file_path=self.file_path).get_file()[:self.n_inference_samples]

        # Load the model and tokenizer
        model, tokenizer = load_model(
            model_id = self.model_id,
            quantization = self.quantization,
            device = self.device,
            hf_auth = self.hf_auth
        ).load()

        # Initialize inference and generate results
        infer = inference(
            data = data,
            truncation = self.truncation,
            padding = self.padding,
            max_length = self.max_length,
            max_new_tokens = self.max_new_tokens,
            tokenizer = tokenizer,
            model = model,
            device = self.device,
            input_col = self.input_col,
            prompt_instruction = self.prompt_instruction,
            few_shot_examples = self.few_shot_examples
        )
        results = infer.generate_results()
        return results