using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.Extensions.Configuration;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.Spark.Sql;
using static Microsoft.Spark.Sql.Functions;
using System.Data.SqlClient;
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
            // Define source data directory paths
            string solutionDirectory = "/home/lqdev/Development/RestaurantInspectionsSparkMLNET";
            // string dataLocation = Path.Combine(solutionDirectory,"RestaurantInspectionsETL","Output");

            // var latestOutput = 
            //     Directory
            //         .GetDirectories(dataLocation)
            //         .Select(directory => new DirectoryInfo(directory))
            //         .OrderByDescending(directoryInfo => directoryInfo.Name)
            //         .Select(directory => directory.FullName)
            //         .First();

            var sc =
                SparkSession
                    .Builder()
                    .AppName("Restaurant_Inspections_Enrichment")
                    .GetOrCreate();

            // var schema = @"
            //     INSPECTIONTYPE string,
            //     CODES string,
            //     CRITICALFLAG int,
            //     INSPECTIONSCORE int,
            //     GRADE string";

            // DataFrame df = 
            //     sc
            //     .Read()
            //     .Schema(schema)
            //     .Csv(Path.Join(latestOutput,"Ungraded"));

            DatabaseLoader loader = _mlContext.Data.CreateDatabaseLoader<DBInput>();
            string sqlCommand = "SELECT * FROM UngradedInspections";
            DatabaseSource dbSource = new DatabaseSource(SqlClientFactory.Instance, _config["connectionString"], sqlCommand);
            IDataView dbData = loader.Load(dbSource);

            IEnumerable<DBInput> dbDataEnumerable = _mlContext.Data.CreateEnumerable<DBInput>(dbData, reuseRowObject: true);

            IEnumerable<ModelInput> modelData =
                dbDataEnumerable
                    .Select(dbInput =>
                    {
                        return new ModelInput
                        {
                            InspectionType = dbInput.InspectionType,
                            Codes = dbInput.Codes,
                            CriticalFlag = (float)dbInput.CriticalFlag,
                            InspectionScore = (float)dbInput.Score,
                            Grade = dbInput.Grade
                        };
                    });

            IDataView data = _mlContext.Data.LoadFromEnumerable(modelData);

            sc.Udf().Register<string, string, int, int, string>("PredictGrade", PredictGrade);

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
            string enrichedOutputPath = Path.Join(solutionDirectory, "RestaurantInspectionsEnrichment", "Output");
            string savePath = Path.Join(enrichedOutputPath, outputId);

            if (!Directory.Exists(savePath))
            {
                Directory.CreateDirectory(enrichedOutputPath);
            }

            enrichedDf.Write().Mode(SaveMode.Overwrite).Csv(savePath);

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

    class DBInput
    {
        public string InspectionType { get; set; }

        public string Codes { get; set; }

        public int CriticalFlag { get; set; }

        public int Score { get; set; }

        public string Grade { get; set; }
    }
}
