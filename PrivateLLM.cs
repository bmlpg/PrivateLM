using System.Diagnostics;
using System.Runtime.InteropServices;
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

        private static bool initialized = false;

        public PrivateLLM(ILogger logger)
        {
            _logger = logger;

            if (initialized)
            {
                return;
            }

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

            initialized = true;
        }

        [DllImport("libc")]
        public static extern int malloc_trim(uint pad);

        public Response Call(
            string SystemPrompt,
            string UserPrompt,
            float Temperature = 0.1f,
            int MaxTokens = 256,
            float TopP = 0.1f,
            string ModelFileURL = "",
            string HFAccessToken = "",
            int ContextSize = 1024,
            int Threads = 1
        )
        {
            // Re-use BulkCall logic for a single item to ensure identical behavior

            var singleRequest = new Request
            {
                SystemPrompt = SystemPrompt,
                UserPrompt = UserPrompt,
                Temperature = Temperature,
                MaxTokens = MaxTokens,
                TopP = TopP
            };

            return BulkCall(new List<Request> { singleRequest }, ModelFileURL, HFAccessToken, ContextSize, Threads).FirstOrDefault();
        }

        public List<Response> BulkCall(
            List<Request> Requests,
            string ModelFileURL = "",
            string HFAccessToken = "",
            int ContextSize = 1024,
            int Threads = 1
        )
        {
            if(ContextSize < 128)
            {
                throw new ArgumentException("ContextSize must be greater or equal than 128");
            }

            if(Threads < 1)
            {
                throw new ArgumentException("Threads must be greater or equal than 1");
            }

            
            var results = new List<Response>();

            var modelPath = Path.Combine(Path.GetTempPath(), "model.gguf");
            var parameters = new ModelParams(modelPath) {
                ContextSize = (uint?)ContextSize,
                GpuLayerCount = 0,
                Threads = Threads,
                UseMemorymap = false
            };

            try
            {
                var weights = WeightsSingleton.LoadModel(parameters, ModelFileURL ,HFAccessToken);
                var executor = new StatelessExecutor(weights, parameters);
                var transformer = new PromptTemplateTransformer(weights, true);

                foreach (var req in Requests)
                {
                    Stopwatch sw = Stopwatch.StartNew();
                    string result = ExecuteInference(executor, transformer, req);
                    sw.Stop();
                    var process = Process.GetCurrentProcess();
                    process.Refresh();

                    results.Add(new Response
                    {
                        Result = result,
                        Duration = sw.ElapsedMilliseconds,
                        TotalMemoryMB = (int)(process.WorkingSet64 / (1024 * 1024))
                    });
                }
            }
            finally
            {
                try
                {
                    malloc_trim(0);
                }
                catch
                {
                    /* Fallback for non-linux environments */
                }
            }

            return results;
        }

        private string ExecuteInference(StatelessExecutor executor, PromptTemplateTransformer transformer, Request request)
        {
            if(request.Temperature < 0.0f || request.Temperature > 2.0f)
            {
                throw new ArgumentException("Temperature must be between 0.0 and 2.0");
            }

            if(request.MaxTokens < 128)
            {
                throw new ArgumentException("MaxTokens must be greated or equal than 128");
            }

            if(request.TopP <= 0.0f || request.TopP > 1.0f)
            {
                throw new ArgumentException("TopP must be between 0.0 (exclusive) and 1.0");
            }

            var history = new ChatHistory();
            history.AddMessage(AuthorRole.System, request.SystemPrompt);
            history.AddMessage(AuthorRole.User, request.UserPrompt);

            string fullPrompt = transformer.HistoryToText(history);
            _logger.LogInformation("Executing prompt: " + fullPrompt);

            using var pipeline = new DefaultSamplingPipeline()
            {
                Temperature = request.Temperature,
                TopP = request.TopP,
                RepeatPenalty = 1.1f,
                PenalizeNewline = false
            };

            var inferenceParams = new InferenceParams()
            {
                MaxTokens = request.MaxTokens,
                AntiPrompts = new List<string> { "<|im_end|>", "<|eot_id|>", "<|end_of_text|>", "<start_of_turn>", "User:", "\n\n\n" },                
                SamplingPipeline = pipeline
            };

            var responseBuilder = new System.Text.StringBuilder();

            Task.Run(async () =>
            {
                await foreach (var token in executor.InferAsync(fullPrompt, inferenceParams))
                {
                    responseBuilder.Append(token);
                }
            }).GetAwaiter().GetResult();

            return responseBuilder.ToString().Trim(); ;
        }

        public void Ping()
        {
        }

    }
}
