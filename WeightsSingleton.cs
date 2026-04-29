using System.Net.Http.Headers;
using LLama;
using LLama.Common;

namespace PrivateLLM
{
    internal class WeightsSingleton
    {
        private static LLamaWeights? instance;
        private static string currentModelFileURL = "";

        public static LLamaWeights LoadModel(ModelParams parameters, string modelFileURL, string hfAccessToken)
        {
            if (currentModelFileURL != modelFileURL)
            {
                instance?.Dispose();
                DownloadModel(modelFileURL, hfAccessToken);
                instance = LLamaWeights.LoadFromFile(parameters);
                currentModelFileURL = modelFileURL;

            }
            return instance;
        }

        private static void DownloadModel(string modelFileURL, string hfAccessToken)
        {
            if (string.IsNullOrWhiteSpace(modelFileURL))
            {
                modelFileURL = "https://huggingface.co/bartowski/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/Qwen2.5-0.5B-Instruct-Q6_K_L.gguf";
            }

            string tempFolder = Path.GetTempPath();
            string modelPath = Path.Combine(tempFolder, "model.gguf");
            string tempDownloadPath = modelPath + ".tmp";

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
                }
            }
            catch (Exception)
            {
                if (File.Exists(tempDownloadPath)) File.Delete(tempDownloadPath);
                throw;
            }
        }
    }
}
