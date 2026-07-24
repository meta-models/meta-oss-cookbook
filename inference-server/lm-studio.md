# LM Studio Bionic

Run Muse Glimmer locally with [LM Studio Bionic](https://lmstudio.ai/muse-glimmer), the AI agent for open models. Bionic and Muse Glimmer are a powerful pair, making it easy to run totally local agentic coding, research task, and even work with documents and slides. Running Muse Glimmer on your own hardware in Bionic means no token costs, and complete data privacy.

## Install

Download LM Studio Bionic from [lmstudio.ai](https://lmstudio.ai).

## Use the agent

Open LM Studio Bionic and download Muse Glimmer to use it in a new session. Attach documents or point Bionic at a local folder, and let the Bionic agent work using its built-in tools.

1. Under **Settings > Local Models > Explore**, select Muse Glimmer and download the recommended model file for your hardware.
2. Create a new project and optionally toggle on **Allow Coding** to let Bionic run commands in your local folders.
3. Select Muse Glimmer from the model picker.
4. Prompt Bionic to complete a task, such as editing attached documents or researching a given topic.

Confirm the model is working by running an agentic task in Bionic. For coding, ask Muse Glimmer to inspect a real repository file, explain the relevant code path, and propose a diff. For non-coding tasks, attach a document and ask Muse Glimmer to extract facts with file citations. 

## Serve

LM Studio ships with an OpenAI compatible local API server.

#### Terminal

Run:

```
lms server start
```

The server starts on port 1234 by default. You can change the port with the `--port` flag.

#### UI

1. Under **Settings -> Local Model API**, toggle on the Local API server.
2. Copy the Base URL to use as the local model endpoint in your scripts, projects, or OpenAI-compatible client.

## Verify tool calling

**In the Bionic app:** Run a task that requires a tool. Tool calls appear in the chat logs as the agent works, providing visibility into which tools are called.

**Through an API:** Send a request that makes one or more tools available and asks the model to use one. Confirm that the response requests the expected tool with valid arguments in the format defined by that API.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Muse Glimmer does not appear in the model picker | The model is not downloaded or the LM Link device is disconnected | Download Muse Glimmer under **Settings -> Explore**. For a remote model, confirm the LM Link device is connected and the model is visible. |
| The model fails to load or runs out of memory | Another model is loaded, the context is too large, or the system lacks available memory | Unload other models and lower the context length. Q4_K_M measured about 24.1 GiB at one request and 16K context, so leave additional memory for the OS and Bionic. |
| Bionic can read files but cannot run commands | **Allow Coding** is disabled for the project | Enable **Allow Coding** and retry the task. |
| `curl` cannot connect to the Local Model API | Bionic is closed, the Local Model API is disabled, or the request uses the wrong port | Open Bionic, enable **Settings -> Local Model API**, and use the displayed Base URL. |
| The Local Model API returns `model not found` | The request uses a name that differs from Bionic's served model ID | Query `GET /v1/models` and copy the exact model ID into the request. |

## Next steps

- Visit [Bionic Docs](https://lmstudio.ai/docs/bionic) to learn more about projects and Bionic features.
- Read LM Studio's [Muse Glimmer blog post](https://lmstudio.ai/blog/muse-glimmer) for example-rich use cases with the model.
