using System;
using System.Diagnostics;
using System.Net.Http.Headers;
using System.Text;
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

        private static string currentModelFileURL = "";

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
                response.Result = Task.Run(async () =>
                {
                    var modelPath = Path.Combine(Path.GetTempPath(), "model.gguf");
                    var parameters = new ModelParams(modelPath)
                    {
                        ContextSize = 1024,
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

                        int maxTokens = 512;
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

                            if (maxTokens <= 0) { maxTokens = 512; }
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

                        var responseBuilder = new System.Text.StringBuilder();

                        await foreach (var token in executor.InferAsync(fullPrompt, inferenceParams))
                        {
                            responseBuilder.Append(token);
                        }

                        return responseBuilder.ToString().Trim();
                    }
                }).GetAwaiter().GetResult();
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
            if (currentModelFileURL == modelFileURL)
            {
                _logger.LogInformation("Model file URL unchanged, skipping download.");
                return;
            }

            _logger.LogInformation("Starting atomic model download...");

            string filePath = Path.Combine(Path.GetTempPath(), "model.gguf");
            string tempFilePath = filePath + ".tmp";

            if (File.Exists(filePath)) File.Delete(filePath);

            try
            {
                HttpClient client = new HttpClient();

                using var request = new HttpRequestMessage(HttpMethod.Get, modelFileURL);
                request.Headers.Add("User-Agent", "OutSystems-PrivateLLM-Plugin-ODC");

                if (!string.IsNullOrWhiteSpace(hfAccessToken))
                {
                    request.Headers.Authorization = new AuthenticationHeaderValue("Bearer", hfAccessToken);
                }

                using var response = client.SendAsync(request, HttpCompletionOption.ResponseHeadersRead).GetAwaiter().GetResult();
                response.EnsureSuccessStatusCode();

                using (var networkStream = response.Content.ReadAsStreamAsync().GetAwaiter().GetResult())
                using (var fileStream = new FileStream(tempFilePath, FileMode.Create, FileAccess.Write, FileShare.None))
                {
                    networkStream.CopyTo(fileStream);
                }

                File.Move(tempFilePath, filePath);

                currentModelFileURL = modelFileURL;

                _logger.LogInformation("Model download complete and verified.");
            }
            catch (Exception ex)
            {
                _logger.LogError($"Download failed: {ex.Message}");
                // Cleanup temp file on failure so the next attempt starts fresh
                if (File.Exists(tempFilePath)) File.Delete(tempFilePath);
                throw;
            }
        }
    }
}
