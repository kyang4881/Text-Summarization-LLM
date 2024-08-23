import pandas as pd
import json
import os
import shutil
from Pipeline import pipeline 

class LLMModel:
    def __init__(self):
        self.output_file_paths = []
        self.files = []
        self.output_file_path = None
        self.file_name = None

    def load_doc(self, document_path):
        data = pd.read_csv(document_path, sep=';', encoding='ISO-8859-1')
        return data

    def process_file(self, File):
        directory = './flagged'
        try:
            shutil.rmtree(directory)
            print("Deleted directory:", directory)
        except Exception as e:
            print("Failed to delete directory:", directory, e)

        self.file_name = os.path.basename(File)
        output_file_path = os.path.join(os.getcwd(), self.file_name)
        shutil.copyfile(File.name, output_file_path)

        if output_file_path not in self.output_file_paths:
            self.files.append(self.file_name)
            self.output_file_paths.append(output_file_path)
            msg = f"File: < {self.file_name} > uploaded./n/nLoaded files:/n{self.files}"
        else:
            self.files.remove(self.file_name)
            self.output_file_paths.remove(output_file_path)
            self.files.append(self.file_name)
            self.output_file_paths.append(output_file_path)
            msg = f"File: < {self.file_name} > already uploaded./n/nLoaded files:/n{self.files}"

        pl = pipeline(
            n_inference_samples=1,
            file_path=output_file_path,
            #model_id='meta-llama/Llama-2-7b-chat-hf',
            model_id="google/flan-t5-base",
            quantization=False,
            device="cpu",
            hf_auth=os.getenv("HUGGING_FACE_API"),
            truncation=True,
            padding=False,
            max_length=128,
            max_new_tokens=256,
            input_col="message",
            prompt_instruction="The following text contains conversational data. Summarize the text in 2-3 sentences.",
            few_shot_examples={}
        )
        results = pl.run()

        output = []
        for index, row in results.iterrows():
            output.append(f"===========================  Conversation ID {row['conversation_id']}  ===========================")
            output.append(f"\nMessage:\n\n{row['message']}\n")
            output.append(f"Summary (prediction):\n\n{row['preds']}\n\n")
        
        return "\n".join(output)
        

if __name__ == "__main__":
    with open('.config.json') as f:
        config_data = json.load(f)

    os.environ["HUGGING_FACE_API"] = config_data["HUGGING_FACE_API"]
    os.environ["OPENAI_API_KEY"] = config_data["OPENAI_API_KEY"]