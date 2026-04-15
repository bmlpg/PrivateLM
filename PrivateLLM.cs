using System.Diagnostics;
using System.Net.Http.Headers;
using LLama;
using LLama.Common;
using LLama.Sampling;
using LLama.Transformers;
using Microsoft.Extensions.Logging;

namespace PrivateLLM
{
    public class PrivateLLM : IPrivateLLM
    {
        private readonly ILogger _logger;

        public PrivateLLM(ILogger logger)
        {
            _logger = logger;

            LLama.Native.NativeLibraryConfig.All.WithLogCallback((level, message) =>
            {
                bool isError = level == LLama.Native.LLamaLogLevel.Error;
                bool isWarning = level == LLama.Native.LLamaLogLevel.Warning;

                if (!isError && !isWarning) return;

                using var activity = Activity.Current?.Source.StartActivity("PrivateLLM.NativeLog");

                if (isError)
                {
                    _logger.LogError($"LL# NATIVE ERROR: {message}");
                    activity?.SetStatus(ActivityStatusCode.Error, message);
                }
                else
                {
                    _logger.LogWarning($"LL# NATIVE WARNING: {message}");
                }
            });
        }
        public Response Call(
            string UserPrompt,
            string SystemPrompt = "You are a helpful assistant.",
            string ModelFileURL = "",
            string HFAccessToken = "",
            bool UseCustomParameters = false,
            CustomParameters? CustomParameters = null
        )
        {
            Stopwatch sw = Stopwatch.StartNew();

            if (string.IsNullOrWhiteSpace(ModelFileURL))
            {
                ModelFileURL = "https://huggingface.co/bartowski/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/Qwen2.5-0.5B-Instruct-Q4_K_S.gguf";
            }

            SetupModel(ModelFileURL, HFAccessToken);

            Response response = new Response();

            try
            {
                var modelPath = Path.Combine(Path.GetTempPath(), "model.gguf");
                var parameters = new ModelParams(modelPath)
                {
                    ContextSize = 512,
                    UseMemorymap = true,
                    UseMemoryLock = false,
                    GpuLayerCount = 0,
                    Threads = 1
                };

                using (var weights = LLamaWeights.LoadFromFile(parameters))
                {
                    var executor = new StatelessExecutor(weights, parameters);

                    var history = new ChatHistory();
                    history.AddMessage(AuthorRole.System, SystemPrompt);
                    history.AddMessage(AuthorRole.User, UserPrompt);
                    var transformer = new PromptTemplateTransformer(weights, true);
                    string fullPrompt = transformer.HistoryToText(history);

                    int maxTokens = 256;
                    float temperature = 0.1f;
                    float topP = 0.9f;

                    if (UseCustomParameters)
                    {
                        if (CustomParameters != null)
                        {
                            maxTokens = CustomParameters.Value.MaxTokens;
                            temperature = (float)CustomParameters.Value.Temperature;
                            topP = (float)CustomParameters.Value.TopP;
                        }

                        if (maxTokens <= 0) { maxTokens = 256; }
                        if (topP <= 0) { topP = 1.0f; }
                    }

                    var inferenceParams = new InferenceParams()
                    {
                        MaxTokens = maxTokens,
                        // THE UNIVERSAL STOP LIST
                        AntiPrompts = new List<string>
                            {
                            "<|im_end|>",    // Qwen, SmolLM2, Phi-3/4
                            "<|eot_id|>",    // Llama 3, 3.1, 3.2
                            "<|end_of_text|>", // Standard fallback
                            "<start_of_turn>", // Gemma 3
                            "User:",           // Generic legacy fallback
                            "\n\n\n"           // 'The Emergency Brake' (Prevents rambling)
                            },
                        SamplingPipeline = new DefaultSamplingPipeline()
                        {
                            Temperature = temperature,
                            TopP = topP,
                            RepeatPenalty = 1.1f,
                            PenalizeNewline = false,

                        }
                    };

                    response.Result = Task.Run(async () =>
                    {
                        var responseBuilder = new System.Text.StringBuilder();

                        await foreach (var token in executor.InferAsync(fullPrompt, inferenceParams))
                        {
                            responseBuilder.Append(token);
                        }

                        return responseBuilder.ToString().Trim();
                    }).GetAwaiter().GetResult();

                }

            }
            finally
            {
                sw.Stop();
                response.Duration = sw.ElapsedMilliseconds;

                GC.Collect();
                GC.WaitForPendingFinalizers();
            }

            return response;
        }

        public void Ping()
        {
        }

        private void SetupModel(string modelFileURL, string hfAccessToken)
        {
            string tempFolder = Path.GetTempPath();
            string modelPath = Path.Combine(tempFolder, "model.gguf");
            string versionFilePath = Path.Combine(tempFolder, "model.url.txt"); // Our "receipt" file
            string tempDownloadPath = modelPath + ".tmp";

            // 1. Read the "receipt" from disk to see what we actually have in Temp
            string storedUrl = "";
            if (File.Exists(versionFilePath) && File.Exists(modelPath))
            {
                storedUrl = File.ReadAllText(versionFilePath).Trim();
            }

            // 2. Compare against the requested URL
            if (storedUrl == modelFileURL)
            {
                _logger.LogInformation("Model on disk matches requested URL. Skipping download.");
                return;
            }

            _logger.LogInformation("Model mismatch or missing. Starting atomic download...");

            // 3. Clean up old files if they exist
            if (File.Exists(modelPath)) File.Delete(modelPath);
            if (File.Exists(tempDownloadPath)) File.Delete(tempDownloadPath);

            try
            {
                using (HttpClient client = new HttpClient())
                {
                    using var request = new HttpRequestMessage(HttpMethod.Get, modelFileURL);
                    request.Headers.Add("User-Agent", "OutSystems-PrivateLLM-Plugin-ODC");

                    if (!string.IsNullOrWhiteSpace(hfAccessToken))
                    {
                        request.Headers.Authorization = new AuthenticationHeaderValue("Bearer", hfAccessToken);
                    }

                    using var response = client.SendAsync(request, HttpCompletionOption.ResponseHeadersRead).GetAwaiter().GetResult();
                    response.EnsureSuccessStatusCode();

                    using (var networkStream = response.Content.ReadAsStreamAsync().GetAwaiter().GetResult())
                    using (var fileStream = new FileStream(tempDownloadPath, FileMode.Create, FileAccess.Write, FileShare.None))
                    {
                        networkStream.CopyTo(fileStream);
                    }

                    // Move the model into place
                    File.Move(tempDownloadPath, modelPath);

                    // 4. Update the "receipt" file so the next call knows what this file is
                    File.WriteAllText(versionFilePath, modelFileURL);

                    _logger.LogInformation("Model download complete and version receipt updated.");
                }
            }
            catch (Exception ex)
            {
                _logger.LogError($"Download failed: {ex.Message}");
                if (File.Exists(tempDownloadPath)) File.Delete(tempDownloadPath);
                if (File.Exists(versionFilePath)) File.Delete(versionFilePath);
                throw;
            }
        }
    }
}
