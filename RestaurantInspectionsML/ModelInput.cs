using Microsoft.ML.Data;

namespace RestaurantInspectionsML
{
    public class ModelInput
    {
        [LoadColumn(0)]
        public string InspectionType { get; set; }

        [LoadColumn(1)]
        public string Codes { get; set; }

        [LoadColumn(2)]
        public float CriticalFlag { get; set; }
        
        [LoadColumn(3)]
        public float InspectionScore { get; set; }
        
        [LoadColumn(4)]
        [ColumnName("Label")]
        public string Grade { get; set; }   
    }
}