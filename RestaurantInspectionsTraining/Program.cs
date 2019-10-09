using System;
using System.Collections.Generic;
using System.Data.SqlClient;
using System.IO;
using System.Linq;
using Microsoft.Extensions.Configuration;
using Microsoft.ML;
using Microsoft.ML.Data;
using static Microsoft.ML.DataOperationsCatalog;
using Microsoft.ML.AutoML;
using RestaurantInspectionsML;

namespace RestaurantInspectionsTraining
{
    class Program
    {
        private static readonly IConfiguration _config;

        static Program()
        {
            _config = new ConfigurationBuilder()
                .SetBasePath(Directory.GetCurrentDirectory())
                .AddJsonFile("appsettings.json", true)
                .Build();
        }
        static void Main(string[] args)
        {
            // Define source data directory paths
            string solutionDirectory = "/home/lqdev/Development/RestaurantInspectionsSparkMLNET";

            // Initialize MLContext
            MLContext mlContext = new MLContext();

            // Load Data from database
            DatabaseLoader loader = mlContext.Data.CreateDatabaseLoader<DBInput>();
            string sqlCommand = "SELECT * FROM GradedInspections";
            DatabaseSource dbSource = new DatabaseSource(SqlClientFactory.Instance,_config["connectionString"],sqlCommand);
            IDataView dbData = loader.Load(dbSource);

            // Map DTO to Entity
            IEnumerable<DBInput> dbDataEnumerable = mlContext.Data.CreateEnumerable<DBInput>(dbData,reuseRowObject:true);
            IEnumerable<ModelInput> modelData = 
                dbDataEnumerable
                    .Select(dbInput => {
                        return new ModelInput
                        {
                            InspectionType = dbInput.InspectionType,
                            Codes=dbInput.Codes,
                            CriticalFlag=(float)dbInput.CriticalFlag,
                            InspectionScore = (float) dbInput.Score,
                            Grade=dbInput.Grade
                        };
                    });

            // Load the data
            IDataView data = mlContext.Data.LoadFromEnumerable(modelData);

            // Split the data
            TrainTestData dataSplit = mlContext.Data.TrainTestSplit(data, testFraction: 0.2);
            IDataView trainData = dataSplit.TrainSet;
            IDataView testData = dataSplit.TestSet;

            // Define experiment settings
            var experimentSettings = new MulticlassExperimentSettings();
            experimentSettings.MaxExperimentTimeInSeconds = 600;
            experimentSettings.OptimizingMetric = MulticlassClassificationMetric.LogLoss;

            // Create experiment
            var experiment = mlContext.Auto().CreateMulticlassClassificationExperiment(experimentSettings);

            // Run experiment
            var experimentResults = experiment.Execute(data, progressHandler: new ProgressHandler());

            // Best Run Results
            var bestModel = experimentResults.BestRun.Model;

            // Evaluate Model
            IDataView scoredTestData = bestModel.Transform(testData);
            var metrics = mlContext.MulticlassClassification.Evaluate(scoredTestData);
            Console.WriteLine($"MicroAccuracy: {metrics.MicroAccuracy}");

            // Save Model
            string modelSavePath = Path.Join(solutionDirectory, "RestaurantInspectionsML", "model.zip");
            mlContext.Model.Save(bestModel, data.Schema, modelSavePath);
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
