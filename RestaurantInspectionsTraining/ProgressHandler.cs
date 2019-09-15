using System;
using Microsoft.ML.Data;
using Microsoft.ML.AutoML;

namespace RestaurantInspectionsTraining
{
    public class ProgressHandler : IProgress<RunDetail<MulticlassClassificationMetrics>>
    {
        public void Report(RunDetail<MulticlassClassificationMetrics> run)
        {
            Console.WriteLine($"Trained {run.TrainerName} with Log Loss {run.ValidationMetrics.LogLoss:0.####} in {run.RuntimeInSeconds:0.##} seconds");
        }
    }
}