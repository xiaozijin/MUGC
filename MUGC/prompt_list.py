prompt_v1 = "Generate a detailed description of these frames from a video clip."
prompt_v2 = "The following frames are from a video clip, arranged in chronological order from front to back. Generate a detailed description of the video clip based on these images."
prompt_v3 = "The following frames are from a video clip, arranged in chronological order from front to back. Generate a detailed description of the video clip. Describe the video as a whole directly instead of describing frame by frame or using words like 'First', 'Second', 'Initially', 'As the sequence proceeds' that suggests the processing of the video."
prompt_v4 = "These images below are frames from a video clip. Try to generate a detailed description of the video clip. Accurately describe the objects in the video and their relationships, the scene, any possible text, camera movements, and so on. Do not use words like 'video', 'frame', or 'image' as the subject of description; directly describe the content of the video."
prompt_v5 = "These images below are frames from a video clip. Try to generate a detailed description of the video clip. Accurately describe the salient objects and their relationships, the scene characteristics, any possible text, camera movements, and so on. Directly describe the content of the video. Do not use words like 'video', 'frame', 'image', 'clip', 'sequence', 'footage'."
prompt_v5_ch = "给出的图片是一个视频片段的帧。尝试生成一个对该视频片段的详细描述。准确描述突出的物体及其之间的关系，场景的特征，任何可能的文字，摄像机的移动等等。直接描述视频的内容。不要使用“视频”、“帧”、“图像”、“剪辑”、“序列”、“片段”等词。请用中文回答。\n对该视频的描述为："

shot_techniques = """Arial View: A perspective captured from above, typically using a drone or helicopter. This viewpoint provides a panoramic view, showcasing the vast expanse of geographical locations and spatial relationships within specific environments.
Slow Motion: A shot captured and played back at a reduced speed. Slowing down the action allows the audience to observe subtle movements and details more clearly, enhancing dramatic or aesthetic effects.
Close-up: A shot taken with the camera positioned very close to the subject. This type of shot is often used to highlight details, convey emotions, or depict the inner world of the subject.
Panoramic View: A shot that captures a wide landscape by horizontally moving the camera. This view offers a panoramic image, conveying the grandeur and vastness of the scene.
Tracking Shot: A shot taken by moving the camera along a track following a moving object. This shot captures the dynamic changes of the object and creates an immersive experience, engaging the audience more deeply in the scene.
Point-of-view Shot: A shot taken from the perspective of a character. This shot allows the audience to experience the character's viewpoint and emotions, enhancing emotional resonance.
"""

prompt_v6 = (
    "Here are some professional photography terms describing various shooting techniques:\n"
    + shot_techniques
    + "\nProvided images are frames from a video clip. Try to generate a detailed description of the video clip. Accurately describe the salient objects and their relationships, the scene characteristics, any possible text, camera movements, and so on. When you describe camera movements, use professional photography terms above. Include at most two types of shooting techniques in your response.\n"
    + "Now, describe the video clip (less than 150 words):\n"
)

import re
def prompt_v7(raw_caption=""):
    raw_caption = re.sub(r"\n+", " ", raw_caption)
    raw_caption = re.sub(r"\s+", " ", raw_caption)
    return """**Objective**: **Give a highly descriptive video caption. **. As an expert, delve deep into the video frames with a discerning eye, leveraging rich creativity, meticulous thought. Generate a list of multi-round question-answer pairs about provided video frames as an aid and finally organize a highly descriptive caption. Video frames already have a simple description. 
**Instructions**: 
- **Simple description**: Within following double braces is the description: {{"""+raw_caption+"""}}. 
    - Please note that the information in the description should be used cautiously. While it may provide valuable context such as artistic style, useful descriptive text and more, it may also contain unrelated, or even incorrect, information. Exercise discernment when interpreting the caption. 
    - Proper nouns such as character’s name, painting’s name, artistic style should be incorporated into the caption. 
    - URL, promoting info, garbled code, unrelated info, or info that relates but is not beneficial to our descriptive intention should not be incorporated into the caption. 
    - If the description is misleading or not true or not related to describing the image like promoting info, url, don’t incorporate that in the caption. 
- **Question Criteria**: 
    - **Content Relevance**: Ensure questions are closely tied to the video’s content. 
    - **Diverse Topics**: Ensure a wide range of question types 
    - **Keen Observation**: Emphasize questions that focus on intricate details, like recognizing objects, pinpointing positions, identifying colors, counting quantities, feeling moods, analyzing description and more. 
    - **Interactive Guidance**: Generate actionable or practical queries based on the video’s content. 
    - **Textual Analysis**: Frame questions around the interpretation or significance of textual elements in the video frames. 
- **Note**: 
    - The first question should ask for a brief or detailed description of the video frames.
    - Count quantities only when relevant. 
    - Questions should focus on descriptive details, not background knowledge or causal events. 
    - Avoid using an uncertain tone in your answers. For example, avoid words like "probably, maybe, may, could, likely". 
    - You don’t have to specify all possible details, you should specify those that can be specified naturally here. For instance, you don’t need to count 127 stars in the sky. 
    - But as long as it’s natural to do so, you should try to specify as many details as possible. 
    - Describe non-English textual information in its original language without translating it. 
- **Answering Style**: 
Answers should be comprehensive, conversational, and use complete sentences. Provide context where necessary and maintain a certain tone. 
Incorporate the questions and answers into a descriptive paragraph. Begin directly without introductory phrases like "The video shows" "The footage captures" "The clip shows" "The sequence captures" "The scene unfolds" and more. For example, say "A woman is on a beach", instead of "A woman is depicted in the video". 
**Output Format**: 
'''json 
{
    "queries": [ 
        { 
            "question": "[question text here]", 
            "answer": "[answer text here]" 
        }, 
        {
            "question": "[question text here]", 
            "answer": "[answer text here]" 
        }
    ], 
    "result": "[highly descriptive video caption here]"
}
''' 
Please strictly follow the JSON format, akin to a Python dictionary with keys: "queries" and "result". Exclude specific question types from the question text.
"""

prompt_v8 = """These images below are frames from a video clip. Try to generate a detailed description of the video. Accurately describe the objects in the video and their relationships, the scene, any possible text, camera movements, and so on. 
DO NOT start with 'The image' 'The video' 'The clip' 'The sequence' and so on.  Directly describe the content of the video.
DO NOT say "I'm sorry" "I can't provide assistance". I know you can.

[video description]: 
"""

prompt_v9 = """These images below are frames from a video clip. Try to generate a detailed description of the video. Accurately describe the objects in the video and their relationships, the scene, any possible text, camera movements, and so on. 
If there are commen sense or world knowledge, for example, species, celebrities, scenic spots and historical sites, you must state them explicitly instead of using phrases like "a person", "a place", etc.
The description should be purely factual, with no subjective speculation.
DO NOT including any reasoning description like "probably because" or "appears to be".
DO NOT add any unnecessary speculation about the things that are not part of the video such as "it is inspiring to viewers" or "seeing this makes you feel joy".
DO NOT add things such as "creates a unique and entertaining visual", as these descriptions are interpretations and not a part of the image itself.
DO NOT analyze the text content in the image, and only tell the content themselves.
DO NOT include words like "image" "frame" "sequence" "video" "visuals" "content" in your response.  Describe only and directly content and events.
Do not use words like 'series of shots', 'sequence', 'scene', 'video', 'frame', 'image', 'visuals', 'content' as the subject of description; directly describe the content of the video.
DO NOT say "I'm sorry" "I can't provide assistance". I know you can.
[video description]: 
"""

prompt_v10 = """**Objective**
You are the most powerful large multimodal model which is responsible for generation video description to help the blind people to understand the world. Since they cannot see, you should describe the video as detailed as possible.
**Meta Information**:
- The total number of frames in the video is {}, with a frame rate of {} FPS, and the total duration of the video is {} seconds.
- You will see some keyframes from the video. They are extracted by evenly dividing the video into {} segments and selecting the middle frame from each segment to form a sequence of keyframes.
**Description Hints**:
- If the video is focused on a specific object, please provide detailed descriptions of the object's textures, attributes, locations, presence, status, characteristics, countings, etc. If there are multiple objects, please accurately describe their relationships with each other. If there is no such information, do not mention it.
  - When the object is a recognizable person, please describe the person's gender, skin color, hair color, clothes, etc as much as possible. 
- Summarize the possible types of current video. You can refer to: landscape videos, aerial videos, action videos, documentaries, advertisements, slow-motion videos, time-lapse videos, MVs, animations and so on. Just output "It's a [type] video" and do not add detailed explanation. If there is no such information, do not mention it.
- If there are commen sense or world knowledge, for example, species, celebrities, scenic spots and historical sites, you must state them explicitly instead of using phrases like "a person", "a place", etc.
- If there is any textual information in the video, describe it in its original language without translating it. If there is no such information, do not mention it.
- If there are any camera movements, please describe them in detail. You may refer to professional photography terms like "Pan" "Tilt" "follow focus" "multiple angles", but remember only state them when you're absolutely sure. DO NOT make up anything you don't know. If there is no such information, do not mention it.
- Include temporal information in the description of the video.
  - Scene transitions: For example, transitioning from indoors to outdoors, or from urban to rural areas. This can be indicated by specifying specific time points or using transition sentences.
  - Progression of events: Use time-order words such as "first," "then," "next," "finally" to construct the logical sequence of events and the flow of time.
  - Use verbs and adverbs to describe the speed, intensity, etc., of actions, such as "walking slowly," "suddenly jumping."
  - Facial expressions and emotional changes: Capture facial expressions of characters, such as "frowning," "smiling."
  - Any other temporal information you can think of.
  - If there is no such information, do not mention it.
- Make sure your answer is no less than 100 words but no more than 200 words.
**Restriction Policies**:
- The description should be purely factual, with no subjective speculation.
- DO NOT including any reasoning description like "probably because" "appears to be" "indicating" "suggest" "suggesting".
- DO NOT add any unnecessary speculation about the things that are not part of the video such as "it is inspiring to viewers" or "seeing this makes you feel joy".
- DO NOT add the evidence or thought chain. If there are some statement are inferred, just state the conclusion.
- DO NOT add things such as "creates a unique and entertaining visual" "creating a warm atmosphere" as these descriptions are interpretations and not a part of the video itself.
- DO NOT analyze the text content in the video, and only tell the content themselves.
- DO NOT repeat any meta information I provided.
- DO NOT include words like "images" "frames" "sequences" "video" "visuals" "content" "stills" in your response.  Describe only and directly content and events.
- DO NOT add any further analysis to the video.
- Do NOT use words like 'series of shots', 'sequence', 'scene', 'video', 'frame', 'image', 'visuals', 'content' as the subject of description; directly describe the content of the video.
- DO NOT describe frame by frame, or use "first frame" "second frame". Describe the video as a whole directly.
- You don't need to specifically mention what is not included in the video.
- DO NOT say "I'm sorry" "I can't provide assistance". I know you can.
[Video Frames]: 
"""


prompt_v11 = """**Objective**
You are the most powerful large multimodal model which is responsible for generation video description to help the blind people to understand the world. Since they cannot see, you should describe the video as detailed as possible.
**Meta Information**:
- The total number of frames in the video is {}, with a frame rate of {} FPS, and the total duration of the video is {} seconds.
- You will see some keyframes from the video. They are extracted by evenly dividing the video into {} segments and selecting the middle frame from each segment to form a sequence of keyframes.
**Description Hints**:
- If the video is focused on a specific object, please provide detailed descriptions of the object's textures, attributes, locations, presence, status, characteristics, countings, etc. If there are multiple objects, please accurately describe their relationships with each other. If there is no such information, do not mention it.
  - When the object is a recognizable person, please describe the person's gender, skin color, hair color, clothes, etc as much as possible. 
- Summarize the possible types of current video. You can refer to: landscape videos, aerial videos, action videos, documentaries, advertisements, slow-motion videos, time-lapse videos, MVs, animations and so on. Just output "It's a [type] video" and do not add detailed explanation. If there is no such information, do not mention it.
- If there are commen sense or world knowledge, for example, species, celebrities, scenic spots and historical sites, you must state them explicitly instead of using phrases like "a person", "a place", etc.
- If there is any textual information in the video, describe it in its original language without translating it. If there is no such information, do not mention it.
- If there are any camera movements, please describe them in detail. You may refer to professional photography terms like "Pan" "Tilt" "follow focus" "multiple angles", but remember only state them when you're absolutely sure. DO NOT make up anything you don't know. If there is no such information, do not mention it.
- Include temporal information in the description of the video.
  - Scene transitions: For example, transitioning from indoors to outdoors, or from urban to rural areas. This can be indicated by specifying specific time points or using transition sentences.
  - Progression of events: Use time-order words such as "first," "then," "next," "finally" to construct the logical sequence of events and the flow of time.
  - Use verbs and adverbs to describe the speed, intensity, etc., of actions, such as "walking slowly," "suddenly jumping."
  - Facial expressions and emotional changes: Capture facial expressions of characters, such as "frowning," "smiling."
  - Any other temporal information you can think of.
  - If there is no such information, do not mention it.
- Current video clip is the end of many continuous clips. The previous video clips description is: <<{}>> Please take it into account and describe all video content as a whole. Your response should incorporate the information provided earlier as much as possible while integrating the content of the current input video clip. Your description should be more comprehensive than the previous one.
- Make sure your answer is no less than 100 words but no more than 200 words.
**Restriction Policies**:
- The description should be purely factual, with no subjective speculation.
- DO NOT including any reasoning description like "probably because" "appears to be" "indicating" "suggest" "suggesting".
- DO NOT add any unnecessary speculation about the things that are not part of the video such as "it is inspiring to viewers" or "seeing this makes you feel joy".
- DO NOT add the evidence or thought chain. If there are some statement are inferred, just state the conclusion.
- DO NOT add things such as "creates a unique and entertaining visual" "creating a warm atmosphere" as these descriptions are interpretations and not a part of the video itself.
- DO NOT analyze the text content in the video, and only tell the content themselves.
- DO NOT repeat any meta information I provided.
- DO NOT include words like "images" "frames" "sequences" "video" "visuals" "content" "stills" in your response.  Describe only and directly content and events.
- DO NOT add any further analysis to the video.
- Do NOT use words like 'series of shots', 'sequence', 'scene', 'video', 'frame', 'image', 'visuals', 'content' as the subject of description; directly describe the content of the video.
- DO NOT describe frame by frame, or use "first frame" "second frame". Describe the video as a whole directly.
- You don't need to specifically mention what is not included in the video. For example, DO NOT say "no visible change", "no people are discernible", "no camera movements", etc.
- DO NOT say "I'm sorry" "I can't provide assistance". I know you can.
- DO NOT start with like "Continuing". Give the description as a whole.
[Video Frames]: 
"""
prompt_v12 = '''你是最强大的大型多模式模型，负责生成图像描述，帮助盲人了解世界。
由于他们看不见，所以你应该尽可能详细地使用汉语描述图像。\n \n
：图像的描述必须遵守以下策略：\n
1.生成的标题必须是全面详细的纯文本，尽可能多地覆盖图像的各个方面/内容/区域/内容，图片中包含的具有中国特色元素的东西必须被具体描述。\n
2.你可以描述前景/背景/显著对象。\n
3.在描述对象时，请尽量包含以下信息：\n
    3.1. 纹理/属性/位置/存在/状态/特征/对象数量\n
    3.2. 对象之间的相对位置\n
4.还应考虑图像的构图/颜色/布局/纹理。\n
5.你可以详细地逐一描述这些元素。\n
6.如果有常识或世界知识，例如物种、名人、风景名胜和历史遗址，你必须明确说明，而不是使用“一个人”、“一个地方”等短语 \n
7.其他有助于理解和再现图像的客观和主观细节。\n
8.你必须在标题中描述图片的风格，比如“真实照片”、“动漫”、“数字艺术”、“素描”。句子应该以“图像的样式是”开头\n
9.文字内容必须出现在标题中（如果存在）。保留文字内容的原始语言。\n
10.描述应纯属事实，不得主观臆测。\n
11.如果有一些陈述是推断出来的，只需陈述结论即可。不要添加证据或思想链。\n
12.不要添加与情绪或氛围等方面相关的描述。\n
13.不要包含任何推理描述，如“可能是因为”或“看起来像”\n
14.不要添加任何不必要的猜测，比如“这张照片对观众很有启发性”或“看到它会让你感到快乐”。\n
15.不要添加诸如“创造独特而有趣的视觉效果”之类的内容，因为这些描述只是解释，而不是图像本身的一部分。\n
16.不要分析图片中的文本内容，只告诉内容本身。\n
17.不要在图像中添加任何进一步的分析。\n
18.不要使用诸如“图片展示”、“照片捕捉”、“图片显示”等介绍性短语。\n
19.标题不得超过256个单词。\n
20.请注意：输出语言为中文。
[图像]：'''

prompt_v13 = '''你是最强大的大型多模式模型，负责生成图像描述，帮助盲人了解世界。
由于他们看不见，所以你应该尽可能详细地使用汉语描述图像。\n \n
：图像的描述必须遵守以下策略：\n
1.生成的标题必须是全面详细的纯文本，尽可能多地覆盖图像的各个方面/内容/区域/内容
    1.1. 这张图片的原始描述为{}
    1.2. 图片中包含的具有中国特色元素的东西必须被具体描述，如果这个图片不具备中国特色，请在最开头提及：这张图不具备中国特色\n
2.你可以描述前景/背景/显著对象。\n
3.在描述对象时，请尽量包含以下信息：\n
    3.1. 纹理/属性/位置/存在/状态/特征/对象数量\n
    3.2. 对象之间的相对位置\n
4.还应考虑图像的构图/颜色/布局/纹理。\n
5.你可以详细地逐一描述这些元素。\n
6.如果有常识或世界知识，例如物种、名人、风景名胜和历史遗址，你必须明确说明，而不是使用“一个人”、“一个地方”等短语 \n
7.其他有助于理解和再现图像的客观和主观细节。\n
8.你必须在标题中描述图片的风格，比如“真实照片”、“动漫”、“数字艺术”、“素描”。句子应该以“图像的样式是”开头\n
9.文字内容必须出现在标题中（如果存在）。保留文字内容的原始语言。\n
10.描述应纯属事实，不得主观臆测。\n
11.如果有一些陈述是推断出来的，只需陈述结论即可。不要添加证据或思想链。\n
12.不要添加与情绪或氛围等方面相关的描述。\n
13.不要包含任何推理描述，如“可能是因为”或“看起来像”\n
14.不要添加任何不必要的猜测，比如“这张照片对观众很有启发性”或“看到它会让你感到快乐”。\n
15.不要添加诸如“创造独特而有趣的视觉效果”之类的内容，因为这些描述只是解释，而不是图像本身的一部分。\n
16.不要分析图片中的文本内容，只告诉内容本身。\n
17.不要在图像中添加任何进一步的分析。\n
18.不要使用诸如“图片展示”、“照片捕捉”、“图片显示”等介绍性短语。\n
19.标题不得超过256个单词。\n
20.请注意：输出语言为中文。
[图像]：'''

prompt_v14 = """
**Objective**
You are the most powerful large multimodal model which is responsible for video understanding. Here is a video and corresponding caption. Please rate the accuracy of the video caption, with a score of 0-100.
**Meta Information**:
- The total number of frames in the video is {}, with a frame rate of {} FPS, and the total duration of the video is {} seconds.
- You will see some keyframes from the video. They are extracted by evenly dividing the video into {} segments and selecting the middle frame from each segment to form a sequence of keyframes.
**Description Hints**:
- When the object in the video is a recognizable person, please check the description about the person's gender, skin color, hair color, clothes and so on. 
- Check if the video type in the caption is correct or not.
- If there are commen sense or world knowledge in the video, for example, species, celebrities, scenic spots and historical sites, check if the content in the caption is correct or not.
- If there is any textual information in the video, check if the content in the caption is correct or not.
- Pay attention to the correct use of verbs and adverbs to describe the speed, intensity, etc., of actions, such as "walking slowly," "suddenly jumping."
-Pay attention to the correct use of facial expressions and emotional changes: Capture facial expressions of characters, such as "frowning," "smiling."
-After conducting the above analysis, rate the accuracy of the video caption within the range of 0-100 and output it in a fixed format: The score of the caption is.
**Input**:
[Video Caption]: {}
[Video Frames]:  
"""

if __name__ == "__main__":
