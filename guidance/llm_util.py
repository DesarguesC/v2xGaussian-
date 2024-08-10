import anthropic
import httpx, base64


def encode_image(image_path):
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def forground_partial(name) -> bool:
    return 'car' in name or 'bus' in name or 'truck' in name

def background_patial(name) -> bool:
    return 'sky' in name or 'building' in name or 'light' in name or 'bridge' in name or 'tree' in name

class Claude():
    def __init__(self, engine, api_key, system_prompt, proxy='http://127.0.0.1:7890', max_tokens=50, temperature=0.8):
        self.engine = engine
        self.api_key = api_key
        self.system_prompt = system_prompt + '\nAny irrelevant characters appear in your response is STRICTLY forbidden. '
        self.proxy = proxy
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.client = anthropic.Client(
            api_key = self.api_key,
            proxies = httpx.Proxy(proxy if isinstance(proxy, str) else 'http://127.0.0.1:7890')
        )

    def vehicle_judge_ask(self, name) -> bool:
        name = name.lower()
        if forground_partial(name):
            return True
        elif background_patial(name):
            return False
        try_time = 3
        while try_time > 0:
            try:
                response = self.client.messages.create(
                    model=self.engine,  # "claude-2.1",
                    max_tokens=self.max_tokens,
                    system=self.system_prompt,  # <-- system prompt
                    temperature=self.temperature,
                    messages=[
                        {"role": "user", "content": name}  # <-- user prompt
                    ]
                ).content[-1].text
            except Exception as err:
                print(f'err: {err}')
                try_time -= 1
                continue

            return 'yes' in response.lower()

        # didn't handle with the net work error
        return False



    def pre_cut(self, prompt):
        return prompt[prompt.find('('):]
    def ask(self, question: str):
        response = self.client.messages.create(
            model = self.engine, # "claude-2.1",
            max_tokens = self.max_tokens,
            system = self.system_prompt,  # <-- system prompt
            temperature = self.temperature,
            messages=[
                {"role": "user", "content": question}  # <-- user prompt
            ]
        ).content[-1].text
        if '(' in response:
            response = response[response.find('('):response.rfind(')')+1]
            res = ''
            for char in response:
                if char not in ['[', '$']:
                    res = res + char
            
            return res
        else: return response


class Vision_Claude():
    def __init__(self, engine, api_key, proxy='http://127.0.0.1:7890', max_tokens=300, temperature=0.8):
        self.engine = engine
        self.api_key = api_key
        self.proxy = proxy
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.client = anthropic.Client(
            api_key=self.api_key,
            proxies=httpx.Proxy(proxy if isinstance(proxy, str) else 'http://127.0.0.1:7890')
        )

    def ask(self, question: str, encoded_img):
        response = self.client.messages.create(
            model = self.engine,
            max_tokens = self.max_tokens,
            temperature = self.temperature,
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": encoded_img,
                            },
                        },
                        {
                            "type": "text",
                            "text": question
                        }
                    ]
                }
            ]
        ).content[-1].text
        print(f'\nRaw response from Claude-3: {response}\n')
        res = ''
        for char in response:
            if char != '$':
                res = res + char
        return res[res.find('('):res.rfind(')')+1]


Question = lambda x, y, w, h: f"Now you get an image, and I want to edit this image with an instruction \"{x}\". "\
                              f"What you should do is to arrange a location for \"{y}\". And you should tell me the "\
                              f"location in form of $(x,y,w,h)$, where $x,y$ indicates the coordinates "\
                              f"and $(w,h)$ indicates the width and height. The image sized {(w,h)}. " + \
                              "Note that you are getting a ratio, so you only need to output a ratio too. " if (w,h) == (1,1) else ""\
                              "Any other characters in your output is strictly forbidden. "


static_question = "You are a bounding box generator. I'm giving you a image and a editing prompt. The prompt is to move a target object to another place, "\
                 "such as \"Move the apple under the desk\", \"move the desk to the left\". "\
                 "What you should do is to return a proper bounding box for it. The output should be in the form of $[Name, (X,Y,W,H)]$"\
                 "For instance, you can output $[\"apple\", (200, 300, 20, 30)]$. Your output cannot contain $(0,0,0,0)$ as bounding box. "

def ask_claude_vision(img_encoded, agent, edit_txt, target, img_size):
    w, h, _ = img_size # (w, h, 3)
    # question = Question(edit_txt, target, w, h)
    question = static_question + f"Here\'s the instruction: {edit_txt}"
    response = agent.ask(question, img_encoded)
    return response

def claude_vision_box(opt, target_noun: str, size):
    img_encoded = encode_image(opt.in_dir)
    agent = Vision_Claude(engine=opt.vision_engine, api_key=opt.api_key)

    raw_return = ask_claude_vision(img_encoded, agent, opt.edit_txt, target_noun, size) # "(x, y, w, h)" string return expected
    raw_return = raw_return[raw_return.find('('): raw_return.rfind(')')+1]

    return f'[{target_noun}, {raw_return}]'




vehicle_prompt = "You will receive the name of an object, and you will need to make the following "\
                 "judgment about it: In the driving scenario, from the driver's perspective, does "\
                 "this object belong to a moving vehicle? If yes, answer \"Yes\", otherwise answer "\
                 "\"no\" without any extra characters. Here is a set of examples: "\
                 "<Question> Bus <Answer> yes\n<Question> Truck <Answer> yes\n"\
                 "<Question> road <Answer> no\m<Question> traffic Lights <Answer> no\n"

def get_vehicle_agent(engine='claude-3-haiku-20240307'):
    import pandas as pd
    api_key = str(list(pd.read_csv('./key.csv')['anthropic'])[0])
    return Claude(engine = engine, api_key = api_key, system_prompt=vehicle_prompt)
