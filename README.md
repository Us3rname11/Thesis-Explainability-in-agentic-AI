# Thesis-Explainability-in-agentic-AI

1. main.py
INSTRUCTIONS:
- run the main with argument "--category" and one of the five categories: "business_and_productivity", "travel_and_transportation", "finance", "data" or "entertainment_and_media"
- select inseq attribution method in line 102 (aktuell Attention, bei Saliency oder Gradients ist mein Macbook abgeschmiert, glaube wegen dem RAM)
- select model in line 120 ("Qwen3-0.6B", "Qwen3-1.7B" or "Qwen3-4B" recommended) <-- 4B wäre gut wenns geht, allerdings schaffen die 1.7 modelle die tasks auch recht zuverlässig
- select number of randomized out-of-category tools to be added to model. (when 0, model has only the 20 in-category tools) <-- bei großen Mengen wird vor allem die Attribution sehr langsam. Würde mal mit 10 starten und gucken 


what it does:
- create a dir "out" and a subdirectory with the category name
- use all task data from "data/dataset_combined.json" with the corresponding category (~50 tasks/prompts)
- add tools from custom_tools.py
- run a Hugging Face smolagent LLM CodeAgent to generate a thinking step and a code snippet, based on reasoning and its available tools
- run a inseq attribution model on the model input + fixed output to find token attribution.
- saves the full memory of the agent and the attribution output object as files

2. custom_tools.py
Contains all tools that the agent will have access to.
Tools in don't have a functional part, only a "pass". The rationale beeing: 1. We only examine the first generation step, so we do not rely on the results of the tool/api for the next steps. 2. The LLM only knows the function name, description and arguments durcing inference, so the model believes it is a functional tool.

@tool
def transcribe_meeting_audio(audio_file: str, language: str) -> str:
    """
    Transcribes the audio from a meeting file into text based on a specified language.
    This is useful for converting spoken content from meetings into written form.

    Args:
        audio_file (str): Path or filename of the meeting audio.
        language (str): Language code for transcription.
    """
    pass

3. tool_list_combined.json
shows all tool names and categories that they got connected to

4. dataset_combined contains ~250 tasks(=userprompts), the ground-truth tool and arguments to use.
100 tasks are taken from taskbench https://huggingface.co/datasets/microsoft/Taskbench/viewer/multimedia/test?f%5Bn_tools%5D%5Bmin%5D=1&f%5Bn_tools%5D%5Bmax%5D=2&views%5B%5D=multimedia
150 tasks are self created

