using OutSystems.ExternalLibraries.SDK;

namespace PrivateLLM
{
    [OSStructure(Description = "")]
    public struct Response
    {
        [OSStructureField(DataType = OSDataType.Text, Description = "Result in text.", IsMandatory = false)]
        public string Result;
        [OSStructureField(DataType = OSDataType.LongInteger, Description = "Duration in milliseconds.", IsMandatory = false)]
        public long Duration;
        [OSStructureField(DataType = OSDataType.Integer, Description = "Total memory consumption in MB.", IsMandatory = false)]
        public int TotalMemoryMB;
    }

    [OSStructure(Description = "")]
    public struct Request
    {
        [OSStructureField(DataType = OSDataType.Text, Description = "Sets the persona and constraints for the AI (e.g., \"You are a metadata extractor\").", IsMandatory = true)]
        public string SystemPrompt;
        [OSStructureField(DataType = OSDataType.Text, Description = "The specific task or data to process.", IsMandatory = true)]
        public string UserPrompt;
        [OSStructureField(DataType = OSDataType.Decimal, Description = "Lower (0.0 - 0.2) for precise data extraction. Higher (0.7+) for creative summaries. Default: 0.1.", IsMandatory = false, DefaultValue = "0.1")]
        public float Temperature;
        [OSStructureField(DataType = OSDataType.Integer, Description = "Controls the maximum length of the response. Increase for long summaries; decrease to save processing time. Default: 256.", IsMandatory = false, DefaultValue = "256")]
        public int MaxTokens;
        [OSStructureField(DataType = OSDataType.Decimal, Description = "Limits the model to a 'nucleus' of high-probability words. Lower (e.g., 0.5) to make the output more focused and predictable. Default: 0.9.", IsMandatory = false, DefaultValue = "0.9")]
        public float TopP;
    }
}
