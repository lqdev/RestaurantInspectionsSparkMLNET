cd ./bin/Debug/netcoreapp2.1/ubuntu.18.04-x64/publish

spark-submit --class org.apache.spark.deploy.dotnet.DotnetRunner --master local microsoft-spark-2.4.x-0.4.0.jar dotnet RestaurantInspectionsEnrichment.dll
