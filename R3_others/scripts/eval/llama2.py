from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
from peft import PeftModel
import torch, os
import logging
import sys

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"]="python"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, stop_words:list, tokenizer):
        self.keywords = [torch.LongTensor(tokenizer.encode(w)[-5:]).to(device) for w in stop_words]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for k in self.keywords:
            if len(input_ids[0]) >= len(k) and torch.equal(input_ids[0][-len(k):], k):
                return True
        return False

    
class LlamaInterface:
    def __init__(self, modelpath, peftpath=None, add_lora=False):
        
        self.tokenizer = AutoTokenizer.from_pretrained(modelpath, padding_side="left")
        DEFAULT_PAD_TOKEN = "[PAD]"
        DEFAULT_EOS_TOKEN = "</s>"
        DEFAULT_BOS_TOKEN = "<s>"
        DEFAULT_UNK_TOKEN = "<unk>"
        special_tokens_dict = {}

        self.tokenizer.unk_token = "<unk>"
        self.tokenizer.bos_token = "<s>"
        self.tokenizer.eos_token = "</s>"
        self.tokenizer.pad_token = "[PAD]"
        self.tokenizer.unk_token_id = 0
        self.tokenizer.bos_token_id = 1
        self.tokenizer.eos_token_id = 2
        self.tokenizer.pad_token_id = self.tokenizer.unk_token_id

        if self.tokenizer.pad_token is None:
            special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
        if self.tokenizer.eos_token is None:
            special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
        if self.tokenizer.bos_token is None:
            special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
        if self.tokenizer.unk_token is None:
            special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN


        self.tokenizer.add_special_tokens(special_tokens_dict)

        # fix
        self.model = AutoModelForCausalLM.from_pretrained(
            modelpath,
            load_in_4bit=False,
            device_map="auto",
        )

        if add_lora and peftpath:
            self.model = PeftModel.from_pretrained(
                self.model,
                peftpath,
                torch_dtype=torch.float16,
            )

            
    def llama(self, prompt, temperature=1, max_tokens=512, stop=None):
        # print(len(self.tokenizer))
        encoded_prompt = self.tokenizer(prompt, padding=True, return_tensors="pt").to(device)
        stop_criteria = StoppingCriteriaList([KeywordsStoppingCriteria(stop, self.tokenizer)]) if stop else None
        # print(encoded_prompt)
        self.model.eval()
        with torch.no_grad():
            generated_ids = self.model.generate(
                **encoded_prompt,
                max_new_tokens=max_tokens,
                do_sample=False,
                early_stopping=False,
                num_return_sequences=1,
                temperature=temperature,
                stopping_criteria=stop_criteria,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        # num_beams=1, do_sample=True, num_return_sequences=1,
        
        outputs = generated_ids.tolist()
        responses = []

        # decoded_output = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        for i in range(len(outputs)):
            output = outputs[i]
            response = self.tokenizer.decode(output[len(encoded_prompt['input_ids'][i]):], skip_special_tokens=True)
            responses.append(response)


        return responses


    def generate_responses_from_llama(self, prompt, temperature=1, max_tokens=512, stop=None):
        responses = self.llama(prompt, temperature, max_tokens, stop)
        return responses