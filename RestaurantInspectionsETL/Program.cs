﻿using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.Extensions.Configuration;
using Microsoft.Spark.Sql;
using static Microsoft.Spark.Sql.Functions;

namespace RestaurantInspectionsETL
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
            // Define columns to remove
            string[] dropCols = new string[]
            {
                "CAMIS",
                "CUISINE DESCRIPTION",
                "VIOLATION DESCRIPTION",
                "BORO",
                "BUILDING",
                "STREET",
                "ZIPCODE",
                "PHONE",
                "ACTION",
                "GRADE DATE",
                "RECORD DATE",
                "Latitude",
                "Longitude",
                "Community Board",
                "Council District",
                "Census Tract",
                "BIN",
                "BBL",
                "NTA"
            };

            // Create SparkSession
            var sc =
                SparkSession
                    .Builder()
                    .AppName("Restaurant_Inspections_ETL")
                    .GetOrCreate();

            // Load data
            DataFrame df =
                sc
                .Read()
                .Option("header", "true")
                .Option("inferSchema", "true")
                .Csv("Data/NYC-Restaurant-Inspections.csv");

            //Remove columns and missing values
            DataFrame cleanDf =
                df
                    .Drop(dropCols)
                    .WithColumnRenamed("INSPECTION DATE", "INSPECTIONDATE")
                    .WithColumnRenamed("INSPECTION TYPE", "INSPECTIONTYPE")
                    .WithColumnRenamed("CRITICAL FLAG", "CRITICALFLAG")
                    .WithColumnRenamed("VIOLATION CODE", "VIOLATIONCODE")
                    .Na()
                    .Drop();

            // Encode CRITICAL FLAG column
            DataFrame labeledFlagDf =
                cleanDf
                    .WithColumn("CRITICALFLAG",
                        When(Functions.Col("CRITICALFLAG") == "Y", 1)
                        .Otherwise(0));

            // Aggregate violations by business and inspection
            DataFrame groupedDf =
                labeledFlagDf
                    .GroupBy("DBA", "INSPECTIONDATE", "INSPECTIONTYPE", "CRITICALFLAG", "SCORE", "GRADE")
                    .Agg(Functions.CollectSet(Functions.Col("VIOLATIONCODE")).Alias("CODES"))
                    .Drop("DBA", "INSPECTIONDATE")
                    .WithColumn("CODES", Functions.ArrayJoin(Functions.Col("CODES"), ","))
                    .Select("INSPECTIONTYPE", "CODES", "CRITICALFLAG", "SCORE", "GRADE");

            // Split into graded and ungraded DataFrames
            DataFrame gradedDf =
                groupedDf
                .Filter(
                    Col("GRADE") == "A" |
                    Col("GRADE") == "B" |
                    Col("GRADE") == "C");

            DataFrame ungradedDf =
                groupedDf
                    .Filter(
                        Col("GRADE") != "A" &
                        Col("GRADE") != "B" &
                        Col("GRADE") != "C");

            // Save DataFrames
            var timestamp = ((DateTimeOffset)DateTime.UtcNow).ToUnixTimeSeconds().ToString();

            var saveDirectory = Path.Join("Output", timestamp);

            if (!Directory.Exists(saveDirectory))
            {
                Directory.CreateDirectory(saveDirectory);
            }

            var gradedOptions = GetDbOptions("GradedInspections");
            var ungradedOptions = GetDbOptions("UngradedInspections");

            gradedDf
                .Write()
                .Format("jdbc")
                .Options(gradedOptions)
                .Mode(SaveMode.Overwrite)
                .Save();

            ungradedDf
                .Write()
                .Format("jdbc")
                .Options(ungradedOptions)
                .Mode(SaveMode.Overwrite)
                .Save();
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
