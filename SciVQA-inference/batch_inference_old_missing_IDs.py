import torch
from PIL import Image
import json
from transformers import MllamaForConditionalGeneration, AutoProcessor
import argparse
from datasets import load_dataset
from tqdm import tqdm
from dotenv import load_dotenv
import os
import traceback
from datetime import datetime
from pydantic import BaseModel, Field, HttpUrl, model_validator
from typing import List, Optional

# Enable better CUDA memory handling
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

parser = argparse.ArgumentParser(description="LlamaV-o1 Batch Inference")
parser.add_argument("--image_dir_path", type=str, required=True, help="Path to the images")
parser.add_argument("--data_type", type=str, default='validation', help="train, test, validation")
parser.add_argument("--samples", type=int, help="No of samples from the selected data type")
parser.add_argument("--num_beams", type=int, default=1, help="Number of beams for beam search")
args = parser.parse_args()

missing_ids = ['0384ba4579fd751d34307fcf82745b64', '04062609897e796d72715dbd4fb05ee3', '048c0f6b71dda72605b39e32758d5f32', '04ae26160c043bfb48d58ef509e62a2f', '078ebf70f3d6748a0c408d803eab1570', '07c31f9732ed741f60155795b270ce74', '07cdd25cfedc23fb99a1a539c2744a38', '0930a0eb3fce83b191769c294e2b342c', '095e3b195735fddb4d7460f80371772e', '09c1818cf6b28decf6c3c4cf5f39d4e5', '0abe1992aa3e40e30ebd29357a83880e', '0ac61ded7029174852be9f69e0d48126', '0b142d1aefe74d205a17ae661ac69999', '0bf99552b78a8baeecd6ea3497e93746', '0dc73c79011d56c8366bf98fd60f98d7', '0de9b8454d76c8807d6e57771230fe6d', '0e7ad448ec5506e098d6bc9468c3ce93', '10332c8718a9a42e48d57c61b21ba713', '1107378f71911a46890c543cd3f7c0bb', '11e6222cb760a157d9508c2eaefff8b2', '11fb40ad08ef13c7477b98038267d941', '15c3ac594c6943dbe18e7b414e3c7821', '1619ca2762aa2fdd8bae9d84d45f02b0', '190d2d5376b6960d81aca28472da0b83', '1a00da2efb15687628c8dc806dcc7f52', '1b5cf9b908e12596466824f4ee23976d', '1b6f559e14f2bc50f1a3b8e48773df3a', '1e38d77230298041985568a4c005891c', '20bb1b25fa8586f95fa34d03216e70fb', '215d85b8bdfceb5d1253acdb342fa975', '21a7f3aae83254a65f276895347de60e', '21aae97a5bb19d967e9f518753a51b8c', '223dc3ee5fbe622e90e708941ff140cb', '22bd5aef1799532a08c629286def21af', '25a68cf49a395fe80dbb74522a524478', '2618a20734e05bb3f042f6d987bc1ab6', '282b28e6427eb7ae16c0b53cb621a9bb', '29e228a68a6d1ce3e0e6ff64ab868898', '2a509a8b328eaee41f27fb6e247063b5', '2ad99ae7ecd34aece9c7a434cd77cc1f', '2b544c56b0fa03ca02b4985e3620ea5c', '2c4959b2cbfb2bd2d7e45dc8afc039c6', '2cbc80e462c2ba49b06a45f1d0976880', '2e7b2d03e0f4891dce92bc2a95322a4c', '2ebb6e97f15ba4e41862400f513814d0', '2f9bb13a115c338e0b9271382887aa52', '2fa3ea3499542cc2b7de30054a06a801', '2fcc2d8152903d5c68fbe98e5b0bdbc0', '3098f352d63d57588051cb223a97d117', '3112d6fb131079cb74d5114ac306f786', '319f1b8c9e27cf277c488e38b94d592b', '31bc588efe795087d5957c25b559ad08', '32259325fda4788f0c953cabd85e6c7e', '3284274a2282a348c3d339de80ee2aa5', '32d09f7fd2c225add43a4c97fb597acd', '32d0b3c4a583ee55df99f7ddf6e0f52c', '3370ae4d9deb43d2842cfeb7a0df8737', '33bd06fbd59dfb64a6393cd697c0f53a', '342624f270ad643d8257a2529bbce673', '365f4c665c96aa9e196daa481f63e7e3', '3775d0a9e6395295ff43ae6c7c5355c8', '37bae23e4a0b018950d1e2b565d2b037', '380a33722e898501499c8d1aabd25e9c', '3878ea37e9d6d26d22f0fa6dcfe1147a', '38881bf2f248f0f1b4216293b9d0f4cc', '39ea9367f134e84cda1e7013c8b8faee', '39ffb985c36590805e9bf45beac43589', '3a8429e694e74c145343f357c4fa828c', '3b8ea43d8d0782f2c37f34531fe4ef41', '3da1f4ba964fd4ed94636d4b4bdd3bfa', '3e16cb9e65ade64053e7f6f8a1d00379', '3e3704b22ef8ce8bcb4f9b05153dfc35', '3ea0395cba052d9149fd27e112275d60', '3edb42ed0644f7ad94d1aeaaeb143a5c', '3f92063643ed6efa79de0dd6b13af7a9', '40552859c7f143d6f07b06e5be7f6e36', '4358bfc5bf70257a34c1c53d0086f684', '44544059d98fd160e1f45b2e5b409ead', '448bf695368fee59224f46dedd7af442', '469155f3ad545e101094b22b2ae7e532', '46ba72a922887c95bf9c5a7e3647b077', '4786852a8d09858acc4ec0118326c784', '49a6c6ce1c988a6af075b79bb14d8a1a', '4a98f64eb2c8af0d00240df9187fdffd', '4c379e8d5ffdef52b71f794ec408b0a8', '4dd103d597a162b75d6a1396a62c0f0a', '4e0ab279dcbc293f3578dfd63b4bacb3', '4f3a6fe120fd15440575631d9f25056d', '4ffe8ff81b4940357a7a6251aee812b2', '51eaef57db39f99c440819aea07c54b8', '5260902527948856d2164c012396cd66', '5272f9832fa10609f402cfe10924ce89', '5275c12ea5bbdf1274909b8a28a6ec59', '530b8e146dcd59419ec043ec85814330', '531f3dccc3d10ad3724181b9efd7b0e7', '55252c4d011a8c359e5f0b3a66b71f9f', '582bd1b652604a4f2d0c00fff729a92a', '585c3f5176577d272bde8bfb510b3823', '59bee5b916691eefdc5000563d681f6e', '5aad709a4877983eecb5419d13d45872', '5b1574f5e20beedb36b32b5e05f7abef', '5b54f20d1986be9efb87a57747a5c0e2', '5bb33c4511acfb98e1b54a75b2903b08', '5bb401d02a55a587c1821c41834594dd', '5c605679a08a45ebd767fa5110c3ccc9', '5ec52152ef73a174e41b19de8239d64e', '5ef1dd382f1bfe5eb916460f57100e3b', '669b0d1759aef0fe713e4c806e9a00a9', '67d0797a13b45a04fa475c433a40d3b5', '68472d369f9b9dc3092d45706fa236fe', '6851972a335370179eaade42c0b03221', '6932c0e3f1231170a5f4395577aba7ac', '6c1287736a8b7e64cf2c34303599a61a', '6f739c2487d24b6f48b69c653a097146', '6fc2a734532b5b767f712c675c14f684', '6fe7a4735cfb5521019cac60c39e3f52', '70b4f3b58a1c4af4f93eddbc759cea34', '70f8f3727ebb87fd89bf926d2f74dee8', '713e795820da4fd46d71372819bedf6c', '73880396f31d6ca5e8296848b4e23718', '742c9a987d2157cdf38d62739aba362f', '764c6353c87da9f93cf3b4e83d45f12d', '76663426848ccb8731b63dc4483f10b2', '7710124f8189f7ec3444723e363419c7', '7896152e652dc341755573166bb4fe08', '79eb3d42fcf026c0d1b34572a4d134dc', '7ad3f8dec4270aaa261a0bcac1e86a34', '7cd698233c86e1b103f03d839eeaafcc', '7e045cc686098f3bf59ae4d1aefd1bf6', '7f6d646e84192aca4a46e1fd0f9868e9', '7fcbd728f9817fc7efda2f0ac7baf161', '8124bfec71d193e9d15c2640694cd585', '8588fc59a158ac0db7ebaf39b9d17f64', '85b21f4e23b1c84e5ecb9d29104846fe', '86861e2f5c07d33951f377d2fb818bf9', '8782411dd78675fc3bf73ecb23264c93', '881317e62b363e542b62252c688c79a8', '88cb4d9bba5c1c31d0e09f6fddb26067', '88da336626c67c70f9e5a264c6b40631', '896f0e7a480eb3f49cf9d524f1b67b4a', '8a0e9cb0a3f2394fcb972d93ba4fe575', '8bf821d9c79e4a63673e5a998315c417', '8ca8a2f2f46e01f6c90ab374001811cb', '8cb02732c3a2a1bf1c2d179a1a2fe9e2', '8dc3c1c9cfa6bbfc1f9b84eaba7b61d1', '8ed5e126fb654f033ba84762017714be', '90027bfcae5a1070a5cde143762ff5de', '90332c50755bfd351407e260d0e651d8', '90492c96039dd049126592039743681d', '92acf589c9f76e76bb5cd9cc13eef9ab', '93116f013fda536bfc7bcd59a4f9dd46', '9518648bf5c1fba1be5c430df1c20d17', '95b5f70813f1b60be415ea7a976fe5b3', '961bec12c181749c3c9edb6e00e5c186', '98dd1d9baef4a72919244ff76b2904dd', '9b395db070f9501240d0a18a3e7c1deb', '9b98d4d529331192ec6458bd5539ff6b', '9c9d8ff8412016c57e4045a4a6fbf864', '9cfab3e793fe20f5580952fa8d1ffe89', '9d745950211725e3606b125e738389f6', '9e185af3f5862ade6d53ccfe95b3cc0a', '9ecc873b3ebe0d0ce380f38c34d6de03', '9f0e6d93a24af2c8ae8d780d6e8feb47', 'a08bebe764fcfe089fb481c42dffe553', 'a25f385ea4a6d186835deba13c478dc4', 'a5c5086898190dc31e3d804a022aa5de', 'a62321eda5e6eae92dd8d7e1a47f64b0', 'a642755d690b7ae9068d5ab0f88b7191', 'a759afa9ee3a18911f9b68ade4d315e1', 'a763facec9418bf7778dd0a5c3dde73d', 'a84701536dda7958d3a5f18461918b6e', 'a84bf7a4eb5761beb4b7f33544be52d8', 'a93e73e07f1a5708798ac13167b3925b', 'a947571099337ed52ff5c48da0446049', 'ac89d97c9df115b3f1b1f6f255dfc6e2', 'ad2c45d2510bcaca53421d4e26ec24df', 'ae815dde97f9aa59cee3790145e8fef6', 'afb1c8b17dfb4eef0d32d6e9c60c6972', 'afd4c7a6ff28a5ed236653fc06daf61c', 'b356470a1442b2f3d40f9a178be4a0b5', 'b3e95209ee4c39dfba9aae2253515a51', 'b5119e5a15281623d945754c17601a15', 'b58c17756c3b4da721cabc02fc27eea5', 'b662e9f7a91aff562b1708b0885cef68', 'b69e4bc9f5e97e62ec61caf00b88d24c', 'b8c36e0a384cc0c8bd9cccccd9c09383', 'baaea709706e3cfb7482bcb2e976872f', 'bb6a00e19edf213d24ec54119b04d8c3', 'bbcb35dad2fc126df9d26f71ecb2b462', 'bdfb09a599f35d8f54c203ec8c3f0ab1', 'be0e88c647360f1d7a4556c03e71ccb0', 'be327cfd84be4f739c9ece28f6c51d5a', 'be87c8c3043c0e03a5408cbe7089a2dc', 'bfdd8f1f0f0aa0fa1f64505a85022b5a', 'c0189f06a191a7bd6f3b43d761daf994', 'c10911f624396407a2ecc22f0065f313', 'c363664ceee55c6dc499cca40416d711', 'c3a45761e299f1a6bd6eaaf6232c9e24', 'c5941515af46fd93260fb2c139f32651', 'c70269865337db73576e31a9753c91ee', 'c82d606f842832a52f9889ee27d5e7fe', 'c8b5c877683cfecbd7b67b3552e23962', 'ca710c2f4b991d46a155562150bb7f56', 'cae00849e79ca1ddd0a5a1093301046e', 'cbcba0f0e3ff6ae79e06ea1fcc2bce22', 'cc9c54971f2e5accb167f6b0fdc9f308', 'cceb65f7f2805ebafc266bb654664655', 'cd730e437e3daa074d6c80f6244840a1', 'cde8e3c2eb0f70194ae2384ecb243eaf', 'ced0582bea3d403d5e657f56d3e8f184', 'cf942d7720a371f1fdd32fc8282baff3', 'd0feabce9817d12051335c04b1355a2c', 'd18d72143a29bbf50879e1efd606ad72', 'd21d1919c84c45cdf29cfe9e0c8ece62', 'd2961c5e8f84afaed9696290fa00f150', 'd457b210acd010ca3b1152fe2dcef92b', 'd53ca4495b2f1d0de468aa7323111deb', 'd6f6e78473fbcc8367763818c8cf141b', 'd6fb84f54c8c9f9be1b44dac10c2484a', 'd7d0cb1141ea1e96cd307e64ef977f3e', 'd7d6eb7377118c5f130d49b6438a3070', 'd9c68707320e7905d1c8f312a931531f', 'da4d4dbd67f9d99b7a2bafff3ec698a4', 'da9a103b47f8da8f5e7c2d8289de8962', 'dc26dc7d422792d2e269ed6d62ab252c', 'dca8b716ac8d18990a0c19da17ea75b6', 'dd169ffb3d25ebfe740e6f26e92a7325', 'dd807b04079fc255ce807c75d902072a', 'ddf2f3894016517a1b05aa6d74db88e0', 'dfdc8bb5ad7d6e96c1e1f5dbd185b138', 'e0c2d6bb1b45bd417a72accccc5155a7', 'e0d4076126d68c00676f493a413ca57b', 'e207ff389a9463ff8b96ad9cbc5bee4c', 'e38c452b4dbe941156f306f9c0361674', 'e4d8a80fed7981f63fe54ec9a36ba8b3', 'e57df286ff7159b9ca8eac52a9fac8f0', 'e5a74211f9defe7b688928df65150c22', 'e635b966087488370e019ad12c374052', 'e84cb01bf1392cf8d9200b07320939c4', 'eb593ca953b285025032114f4ba6631f', 'eceec9c374e727788e71c9e3a1227720', 'ed05b4d5c7f7078a474fb7202a450ae7', 'eeaff810d21b90d24541607e353d63f7', 'ef59f000a145d6022b90af76bb01760f', 'f0483e4830c09cb8382540fd5c168966', 'f137398528beb2e744ade5a96918f7d1', 'f168fc4f8354d82b1efd4b4a00b238ae', 'f45e2a1d0d7968f246b148d51f89e831', 'f67b9f35df9815f2183b23234c3d7384', 'f76ab060e12ec07caffb07b15ebc607d', 'f885c07ab0edeaf6e00057379bcc9fd2', 'f9065b1dff30fada5013864c5572548a', 'fb20a8d4c6837cb59f43de2451ff761f', 'fb8fd21e97d3d4596c9d323efd1e6003', 'fcc6692c6b6ed74524e69df0cb8dcb5e', 'fcd04bb546ce81edb6334a984f337279', 'fd919770a0524b066e294f5772c92eaf', 'fe01401e90e9608f449bdc2298f30712', 'fe550c3b18fadde715e80f172dacd787', 'ff618753ac73f835c25232a02684b47a', 'ffb0fd454d6840e6fc827ff6fb86f2d8']

class QAImageData(BaseModel):
    instance_id: str
    image_file: str
    figure_id: str
    caption: str
    figure_type: str
    compound: bool
    figs_numb: str
    qa_pair_type: str
    question: str
    answer: Optional[str]
    answer_options: dict
    venue: str
    categories: str
    source_dataset: str
    paper_id: str
    pdf_url: HttpUrl

    def load_image(self) -> Image.Image:
        return Image.open(os.path.join(args.image_dir_path, self.image_file))

    @model_validator(mode="before")
    def merge_answer_options(cls, values):
        options = values.get('answer_options')
        if isinstance(options, list):
            merged = {}
            for item in options:
                merged.update(item)
            values['answer_options'] = merged
        return values

class SciQVALlamaVO1Inference():

    def __init__(self, model_path='omkarthawakar/LlamaV-o1'):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MllamaForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float32
        ).to(device).eval()

        self.processor = AutoProcessor.from_pretrained(model_path)
        self.device = device
        self.run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        os.makedirs("results", exist_ok=True)

        self.kwargs = {
            'max_new_tokens': 512,
            "top_p": 0.9,
            "pad_token_id": 128004,
            "bos_token_id": 128000,
            "do_sample": True,
            "eos_token_id": [128001, 128008, 128009],
            "temperature": 0.4,
            "num_beams": args.num_beams,
            "use_cache": True,
        }
        self.outputs = []

    def _get_caption_prompt(self, caption):
        return f"Provide a detailed description of the image titled {{{caption}}}, particularly emphasizing the aspects related to the question."

    def _get_summary_prompt(self, question):
        return question + "\nSummarize how you will approach the problem and explain the steps you will take to reach the answer."

    def _get_reasoning_prompt(self):
        return (
            "Provide a chain-of-thought, logical explanation of the problem. "
            "In case of multiple plots, carefully compare the y-axis and x-axis scales across plots and note any differences. "
            "Identify which lines correspond to which entities and distinguish them from shaded regions, which represent confidence intervals. "
            "Pay attention to overlapping lines or narrow peaks. Provide a step-by-step reasoning to reach the answer."
        )

    def _get_conclusion_prompt(self):
        return "Return a short, exact answer, no more than a few words. Do not explain or describe. The answer does not have to be a full sentence."

    def _get_non_binary_qa_pair_prompt(self, answer_options):
        return (
            f"Based on the reasoning above, match it to one or more of the provided answer options: {answer_options}. "
            "Return only the corresponding letter(s) of the correct answer(s). "
            "Do not explain your choice, do not rephrase the answer, and do not repeat the option text. "
            "Only output the letter(s) corresponding to the correct choice. "
            "If multiple letters are correct, separate them by commas without spaces (for example: B,C). "
            "If all options are correct, return A,B,C,D. "
            "Do not add anything else."
        )

    def _get_binary_qa_pair_prompt(self):
        return "Return either 'Yes' or 'No'. Do not add anything else - not even punctuation marks."

    def _get_qa_pair_prompt(self):
        return "Give the exact correct answer, with no extra explanation."

    def generate_inner(self, input: QAImageData):
        def __infer(messages: dict) -> str:
            input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            image = input.load_image().resize((224, 224))
            inputs = self.processor(image, input_text, return_tensors='pt')
            inputs = {
                k: v.to(self.device).float() if v.dtype.is_floating_point else v.to(self.device)
                for k, v in inputs.items()
            }
            torch.cuda.empty_cache()  # try to free any unused memory before inference
            with torch.no_grad():
                output = self.model.generate(**inputs, **self.kwargs)
            torch.cuda.empty_cache()
            return self.processor.decode(output[0][inputs['input_ids'].shape[1]:]).replace('<|eot_id|>', '').replace("<|end_of_text|>", "")

        def __tmp(inp, out):
            return [
                {'role': 'assistant', 'content': [{'type': 'text', 'text': inp}]},
                {'role': 'user', 'content': [{'type': 'text', 'text': out}]}
            ]

        messages = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': self._get_summary_prompt(input.question)}]}]
        summary_qa = __infer(messages)

        caption_prompt = self._get_caption_prompt(input.caption)
        messages.extend(__tmp(summary_qa, caption_prompt))
        summary_qa = __infer(messages)

        reasoning_prompt = self._get_reasoning_prompt()
        messages.extend(__tmp(summary_qa, reasoning_prompt))
        reasoning_qa = __infer(messages)

        conclusion_prompt = self._get_conclusion_prompt()
        messages.extend(__tmp(summary_qa, conclusion_prompt))
        summary_qa = __infer(messages)

        if "closed-ended" in input.qa_pair_type and "finite answer set" in input.qa_pair_type:
            if "non-binary" in input.qa_pair_type and input.answer_options:
                qa_pair_prompt = self._get_non_binary_qa_pair_prompt(input.answer_options)
            elif "binary" in input.qa_pair_type:
                qa_pair_prompt = self._get_binary_qa_pair_prompt()
            else:
                qa_pair_prompt = self._get_qa_pair_prompt()
        else:
            qa_pair_prompt = self._get_qa_pair_prompt()

        messages.extend(__tmp(summary_qa, qa_pair_prompt))
        output = __infer(messages)

        print(f"Question: {input.question}\nAnswer: {output}")
        return output, reasoning_qa

    def _should_override_with_unanswerable(self, reasoning: str) -> bool:
        keywords = ["cannot be determined", "cannot determine", "not possible to determine", "insufficient information", "not enough information", "unanswerable", "data is missing", "lack of information", "figure is not provided"]
        return any(keyword in reasoning.lower() for keyword in keywords)

    def _process_input(self):
        dataset = load_dataset("katebor/SciVQA", split=args.data_type)
        missing_entries = [data for data in dataset if data.get("instance_id") in missing_ids]
        
        for idx, data in enumerate(tqdm(missing_entries)):
            try:
                input = QAImageData(**data)
                result, reasoning = self.generate_inner(input)
                if self._should_override_with_unanswerable(reasoning):
                    result = "It is not possible to answer this question based only on the provided data."
                self.outputs.append({"instance_id": input.instance_id, "question": input.question, "answer_pred": result, "reasoning": reasoning})
                if idx % 10 == 0:
                    with open(f"results/{self.run_name}_sciqva_llamaVo1_{idx}.json", "w") as json_file:
                        json.dump(self.outputs, json_file, indent=4)
                        self.outputs = []
            except Exception as e:
                print(f"Error at index {idx}: {e}")
                traceback.print_exc()
                continue

if __name__ == "__main__":
    SciQVALlamaVO1Inference()._process_input()