spark-submit \
--class org.apache.spark.deploy.dotnet.DotnetRunner \
--master local ./bin/Debug/netcoreapp2.1/ubuntu.18.04-x64/publish/microsoft-spark-2.4.x-0.4.0.jar \
dotnet ./bin/Debug/netcoreapp2.1/ubuntu.18.04-x64/publish/RestaurantInspectionsETL.dll
