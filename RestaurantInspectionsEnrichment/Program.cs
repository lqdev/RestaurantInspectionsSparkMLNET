using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.Spark.Sql;
using static Microsoft.Spark.Sql.Functions;
using RestaurantInspectionsML;

namespace RestaurantInspectionsEnrichment
{
    class Program
    {
        private static readonly PredictionEngine<ModelInput,ModelOutput> _predictionEngine;
        
        static Program()
        {
            MLContext mlContext = new MLContext();
            ITransformer model = mlContext.Model.Load("model.zip",out DataViewSchema schema);
            _predictionEngine = mlContext.Model.CreatePredictionEngine<ModelInput,ModelOutput>(model);
        }

        static void Main(string[] args)
        {
            // Define source data directory paths
            string solutionDirectory = "/home/lqdev/Development/RestaurantInspectionsSparkMLNET";
            string dataLocation = Path.Combine(solutionDirectory,"RestaurantInspectionsETL","Output");

            var latestOutput = 
                Directory
                    .GetDirectories(dataLocation)
                    .Select(directory => new DirectoryInfo(directory))
                    .OrderBy(directoryInfo => directoryInfo.Name)
                    .Select(directory => directory.FullName)
                    .First();

            var sc = 
                SparkSession
                    .Builder()
                    .AppName("Restaurant_Inspections_Enrichment")
                    .GetOrCreate();

            var schema = @"
                INSPECTIONTYPE string,
                CODES string,
                CRITICALFLAG int,
                INSPECTIONSCORE int,
                GRADE string";

            DataFrame df = 
                sc
                .Read()
                .Schema(schema)
                .Csv(Path.Join(latestOutput,"Ungraded"));

            sc.Udf().Register<string,string,int,int,string>("PredictGrade",PredictGrade);

            var enrichedDf = 
                df
                .Select(
                    Col("INSPECTIONTYPE"),
                    Col("CODES"),
                    Col("CRITICALFLAG"),
                    Col("INSPECTIONSCORE"),
                    CallUDF("PredictGrade",
                        Col("INSPECTIONTYPE"),
                        Col("CODES"),
                        Col("CRITICALFLAG"),
                        Col("INSPECTIONSCORE")
                    ).Alias("PREDICTEDGRADE")
                );
            

            string outputId = new DirectoryInfo(latestOutput).Name;
            string enrichedOutputPath = Path.Join(solutionDirectory,"RestaurantInspectionsEnrichment","Output");
            string savePath = Path.Join(enrichedOutputPath,outputId);

            if(!Directory.Exists(savePath))
            {
                Directory.CreateDirectory(enrichedOutputPath);
            }

            enrichedDf.Write().Csv(savePath);

        }

        public static string PredictGrade(
            string inspectionType,
            string violationCodes,
            int criticalFlag,
            int inspectionScore)
        {
            ModelInput input = new ModelInput
            {
                InspectionType=inspectionType,
                Codes=violationCodes,
                CriticalFlag=(float)criticalFlag,
                InspectionScore=(float)inspectionScore
            };

            ModelOutput prediction = _predictionEngine.Predict(input);

            return prediction.PredictedLabel;
        }
    }
}
