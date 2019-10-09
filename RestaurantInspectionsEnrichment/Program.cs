using System.Collections.Generic;
using System.IO;
using Microsoft.Extensions.Configuration;
using Microsoft.ML;
using Microsoft.Spark.Sql;
using static Microsoft.Spark.Sql.Functions;
using RestaurantInspectionsML;

namespace RestaurantInspectionsEnrichment
{
    class Program
    {
        private static readonly IConfiguration _config;
        private static readonly MLContext _mlContext;
        private static readonly PredictionEngine<ModelInput, ModelOutput> _predictionEngine;

        static Program()
        {
            _mlContext = new MLContext();

            ITransformer model = _mlContext.Model.Load("model.zip", out DataViewSchema schema);

            _predictionEngine = _mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(model);

            _config = new ConfigurationBuilder()
                .SetBasePath(Directory.GetCurrentDirectory())
                .AddJsonFile("appsettings.json", true)
                .Build();
        }

        static void Main(string[] args)
        {
            // Initialize Spark Session
            var sc =
                SparkSession
                    .Builder()
                    .AppName("Restaurant_Inspections_Enrichment")
                    .GetOrCreate();

            // Define JDBC options to read from
            var readOptions = GetDbOptions("UngradedInspections"); 
                
            // Load the ungraded inspection data
            DataFrame df = 
                sc
                .Read()
                .Format("jdbc")
                .Options(readOptions)
                .Load();
     
            // Register UDF to make predictions using ML.NET model
            sc.Udf().Register<string, string, int, int, string>("PredictGrade", PredictGrade);

            // Apply PredictGrade UDF
            var enrichedDf =
                df
                .Select(
                    Col("INSPECTIONTYPE"),
                    Col("CODES"),
                    Col("CRITICALFLAG"),
                    Col("SCORE"),
                    CallUDF("PredictGrade",
                        Col("INSPECTIONTYPE"),
                        Col("CODES"),
                        Col("CRITICALFLAG"),
                        Col("SCORE")
                    ).Alias("PREDICTEDGRADE")
                );

            // Save Enriched Data
            var writeOptions = GetDbOptions("EnrichedInspections");

            enrichedDf
                .Write()
                .Format("jdbc")
                .Mode(SaveMode.Overwrite)
                .Options(writeOptions)
                .Save();
        }

        public static string PredictGrade(
            string inspectionType,
            string violationCodes,
            int criticalFlag,
            int inspectionScore)
        {
            ModelInput input = new ModelInput
            {
                InspectionType = inspectionType,
                Codes = violationCodes,
                CriticalFlag = (float)criticalFlag,
                InspectionScore = (float)inspectionScore
            };

            ModelOutput prediction = _predictionEngine.Predict(input);

            return prediction.PredictedLabel;
        }

        static Dictionary<string, string> GetDbOptions(string query)
        {
            return new Dictionary<string, string>()
            {
                {"url",_config["url"]},
                {"dbtable",query},
                {"username", _config["username"]},
                {"password",_config["password"]}
            };
        }
    }
}
