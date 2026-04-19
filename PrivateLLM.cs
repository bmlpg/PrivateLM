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
            string SystemPrompt,
            string UserPrompt,
            bool UseCustomParameters = false,
            CustomParameters? CustomParameters = null,
            string ModelFileURL = "",
            string HFAccessToken = ""
        )
        {
            // Re-use BulkCall logic for a single item to ensure identical behavior

            var singleRequest = new Request
            {
                SystemPrompt = SystemPrompt,
                UserPrompt = UserPrompt,
                CustomParameters = UseCustomParameters ? CustomParameters : null
            };

            return BulkCall(new List<Request> { singleRequest }, ModelFileURL, HFAccessToken).FirstOrDefault();
        }

        public List<Response> BulkCall(
            List<Request> Requests,
            string ModelFileURL = "",
            string HFAccessToken = ""
        )
        {
            DownloadModel(ModelFileURL, HFAccessToken);
            var results = new List<Response>();

            var modelPath = Path.Combine(Path.GetTempPath(), "model.gguf");
            var parameters = new ModelParams(modelPath) { ContextSize = 512, GpuLayerCount = 0, Threads = 1 };

            try
            {
                using var weights = LLamaWeights.LoadFromFile(parameters);
                var executor = new StatelessExecutor(weights, parameters);

                foreach (var req in Requests)
                {
                    Stopwatch sw = Stopwatch.StartNew();
                    string result = ExecuteInference(weights, executor, req);
                    sw.Stop();

                    results.Add(new Response
                    {
                        Result = result,
                        Duration = sw.ElapsedMilliseconds
                    });
                }
            }
            finally
            {
                GC.Collect();
                GC.WaitForPendingFinalizers();
            }

            return results;
        }

        private string ExecuteInference(LLamaWeights weights, StatelessExecutor executor, Request request)
        {
            if ((request.SystemPrompt.Length + request.UserPrompt.Length) > 1600)
            {
                throw new ArgumentException("Input is too long for the current model optimization (Max ~400 words).");
            }

            var history = new ChatHistory();
            history.AddMessage(AuthorRole.System, request.SystemPrompt);
            history.AddMessage(AuthorRole.User, request.UserPrompt);

            var transformer = new PromptTemplateTransformer(weights, true);
            string fullPrompt = transformer.HistoryToText(history);

            int maxTokens = 256;
            float temperature = 0.1f;
            float topP = 0.9f;

            if (request.UseCustomParameters)
            {
                if (request.CustomParameters != null)
                {
                    maxTokens = request.CustomParameters.Value.MaxTokens;
                    temperature = (float)request.CustomParameters.Value.Temperature;
                    topP = (float)request.CustomParameters.Value.TopP;

                    if (maxTokens <= 0) { maxTokens = 256; }
                    if (temperature < 0 || temperature > 1) { temperature = 0.1f; }
                    if (topP < 0 || topP > 1) { topP = 0.9f; }
                }
            }

            // Extract parameters from request or use defaults
            var inferenceParams = new InferenceParams()
            {
                MaxTokens = maxTokens,
                AntiPrompts = new List<string> { "<|im_end|>", "<|eot_id|>", "<|end_of_text|>", "<start_of_turn>", "User:", "\n\n\n" },
                SamplingPipeline = new DefaultSamplingPipeline()
                {
                    Temperature = temperature,
                    TopP = topP,
                    RepeatPenalty = 1.1f,
                    PenalizeNewline = false
                }
            };

            string result = Task.Run(async () =>
            {
                var responseBuilder = new System.Text.StringBuilder();

                await foreach (var token in executor.InferAsync(fullPrompt, inferenceParams))
                {
                    responseBuilder.Append(token);
                }

                return responseBuilder.ToString().Trim();
            }).GetAwaiter().GetResult();

            return result;
        }

        public void Ping()
        {
        }

        private void DownloadModel(string modelFileURL, string hfAccessToken)
        {
            if (string.IsNullOrWhiteSpace(modelFileURL))
            {
                modelFileURL = "https://huggingface.co/bartowski/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/Qwen2.5-0.5B-Instruct-Q4_K_S.gguf";
            }

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
