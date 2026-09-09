using System;
using System.Collections.Generic;
using ZeroGeometry.Core.Spatial;
using ZeroTensor.Core;

namespace ZeroGeometry.Core.PointCloud
{
    public sealed class IcpResult
    {
        public double[,] Rotation { get; } // 3x3
        public Point3D Translation { get; }
        public double FitnessRms { get; }
        public int Iterations { get; }
        public bool Converged { get; }
        public PointCloud3D AlignedCloud { get; }

        public IcpResult(
            double[,] rotation,
            Point3D translation,
            double fitnessRms,
            int iterations,
            bool converged,
            PointCloud3D alignedCloud)
        {
            Rotation = rotation;
            Translation = translation;
            FitnessRms = fitnessRms;
            Iterations = iterations;
            Converged = converged;
            AlignedCloud = alignedCloud;
        }
    }

    /// <summary>
    /// Point-to-Point Iterative Closest Point (ICP) algorithm for rigid 3D point cloud registration.
    /// Finds optimal rotation R and translation t aligning source point cloud to target.
    /// </summary>
    public static class IcpRegistration
    {
        public static IcpResult Align(
            PointCloud3D source,
            PointCloud3D target,
            int maxIterations = 50,
            double tolerance = 1e-5,
            double maxCorrespondenceDistance = double.PositiveInfinity)
        {
            if (source == null) throw new ArgumentNullException(nameof(source));
            if (target == null) throw new ArgumentNullException(nameof(target));
            if (source.Count == 0 || target.Count == 0)
                throw new ArgumentException("Source and target point clouds must not be empty.");

            var targetTree = new KdTree3D(target.Points);

            // Accumulated transformation: R_total = I, t_total = 0
            double[,] R_total = new double[3, 3]
            {
                { 1, 0, 0 },
                { 0, 1, 0 },
                { 0, 0, 1 }
            };
            Point3D t_total = new Point3D(0, 0, 0);

            var currentSourcePoints = new Point3D[source.Count];
            for (int i = 0; i < source.Count; i++) currentSourcePoints[i] = source[i];

            double prevRms = double.MaxValue;
            double currentRms = double.MaxValue;
            bool converged = false;
            int iter = 0;

            for (iter = 0; iter < maxIterations; iter++)
            {
                // Step 1: Find correspondences
                var matchedSource = new List<Point3D>(source.Count);
                var matchedTarget = new List<Point3D>(source.Count);
                double sumDistSq = 0.0;

                for (int i = 0; i < currentSourcePoints.Length; i++)
                {
                    var p = currentSourcePoints[i];
                    if (targetTree.NearestNeighbor(p, out var q, out double distSq, out _))
                    {
                        if (distSq <= maxCorrespondenceDistance * maxCorrespondenceDistance)
                        {
                            matchedSource.Add(p);
                            matchedTarget.Add(q);
                            sumDistSq += distSq;
                        }
                    }
                }

                if (matchedSource.Count < 3)
                    break; // Too few matches

                currentRms = Math.Sqrt(sumDistSq / matchedSource.Count);
                if (Math.Abs(prevRms - currentRms) < tolerance)
                {
                    converged = true;
                    break;
                }
                prevRms = currentRms;

                // Step 2: Compute centroids
                double meanPx = 0, meanPy = 0, meanPz = 0;
                double meanQx = 0, meanQy = 0, meanQz = 0;
                int n = matchedSource.Count;

                for (int i = 0; i < n; i++)
                {
                    meanPx += matchedSource[i].X;
                    meanPy += matchedSource[i].Y;
                    meanPz += matchedSource[i].Z;

                    meanQx += matchedTarget[i].X;
                    meanQy += matchedTarget[i].Y;
                    meanQz += matchedTarget[i].Z;
                }

                var pBar = new Point3D(meanPx / n, meanPy / n, meanPz / n);
                var qBar = new Point3D(meanQx / n, meanQy / n, meanQz / n);

                // Step 3: Compute cross-covariance matrix H = sum((p - pBar) * (q - qBar)^T)
                var H = Tensor.Zeros<float>(3, 3);
                for (int i = 0; i < n; i++)
                {
                    float px = (float)(matchedSource[i].X - pBar.X);
                    float py = (float)(matchedSource[i].Y - pBar.Y);
                    float pz = (float)(matchedSource[i].Z - pBar.Z);

                    float qx = (float)(matchedTarget[i].X - qBar.X);
                    float qy = (float)(matchedTarget[i].Y - qBar.Y);
                    float qz = (float)(matchedTarget[i].Z - qBar.Z);

                    H[0, 0] += px * qx; H[0, 1] += px * qy; H[0, 2] += px * qz;
                    H[1, 0] += py * qx; H[1, 1] += py * qy; H[1, 2] += py * qz;
                    H[2, 0] += pz * qx; H[2, 1] += pz * qy; H[2, 2] += pz * qz;
                }

                // Step 4: SVD(H) = U * S * V^T
                TensorDecompositions.SVD(H, out var U, out var _, out var Vt);
                var V = Vt.Transpose(0, 1);

                // R_step = V * U^T
                var Ut = U.Transpose(0, 1);
                var R_step_tensor = TensorBlas.MatMul(V, Ut);

                // Check reflection: det(R)
                float det = R_step_tensor[0, 0] * (R_step_tensor[1, 1] * R_step_tensor[2, 2] - R_step_tensor[1, 2] * R_step_tensor[2, 1]) -
                            R_step_tensor[0, 1] * (R_step_tensor[1, 0] * R_step_tensor[2, 2] - R_step_tensor[1, 2] * R_step_tensor[2, 0]) +
                            R_step_tensor[0, 2] * (R_step_tensor[1, 0] * R_step_tensor[2, 1] - R_step_tensor[1, 1] * R_step_tensor[2, 0]);

                if (det < 0f)
                {
                    // Negate 3rd column of V
                    V[0, 2] = -V[0, 2];
                    V[1, 2] = -V[1, 2];
                    V[2, 2] = -V[2, 2];
                    R_step_tensor = TensorBlas.MatMul(V, Ut);
                }

                double[,] R_step = new double[3, 3];
                for (int r = 0; r < 3; r++)
                    for (int c = 0; c < 3; c++)
                        R_step[r, c] = R_step_tensor[r, c];

                // t_step = qBar - R_step * pBar
                double t_stepX = qBar.X - (R_step[0, 0] * pBar.X + R_step[0, 1] * pBar.Y + R_step[0, 2] * pBar.Z);
                double t_stepY = qBar.Y - (R_step[1, 0] * pBar.X + R_step[1, 1] * pBar.Y + R_step[1, 2] * pBar.Z);
                double t_stepZ = qBar.Z - (R_step[2, 0] * pBar.X + R_step[2, 1] * pBar.Y + R_step[2, 2] * pBar.Z);
                var t_step = new Point3D(t_stepX, t_stepY, t_stepZ);

                // Step 5: Update current source points
                for (int i = 0; i < currentSourcePoints.Length; i++)
                {
                    var p = currentSourcePoints[i];
                    double nx = R_step[0, 0] * p.X + R_step[0, 1] * p.Y + R_step[0, 2] * p.Z + t_step.X;
                    double ny = R_step[1, 0] * p.X + R_step[1, 1] * p.Y + R_step[1, 2] * p.Z + t_step.Y;
                    double nz = R_step[2, 0] * p.X + R_step[2, 1] * p.Y + R_step[2, 2] * p.Z + t_step.Z;
                    currentSourcePoints[i] = new Point3D(nx, ny, nz);
                }

                // Step 6: Accumulate total transformation
                // R_total = R_step * R_total
                // t_total = R_step * t_total + t_step
                double[,] newR = new double[3, 3];
                for (int r = 0; r < 3; r++)
                {
                    for (int c = 0; c < 3; c++)
                    {
                        newR[r, c] = R_step[r, 0] * R_total[0, c] +
                                     R_step[r, 1] * R_total[1, c] +
                                     R_step[r, 2] * R_total[2, c];
                    }
                }
                R_total = newR;

                double newTx = R_step[0, 0] * t_total.X + R_step[0, 1] * t_total.Y + R_step[0, 2] * t_total.Z + t_step.X;
                double newTy = R_step[1, 0] * t_total.X + R_step[1, 1] * t_total.Y + R_step[1, 2] * t_total.Z + t_step.Y;
                double newTz = R_step[2, 0] * t_total.X + R_step[2, 1] * t_total.Y + R_step[2, 2] * t_total.Z + t_step.Z;
                t_total = new Point3D(newTx, newTy, newTz);
            }

            var alignedCloud = new PointCloud3D(currentSourcePoints);
            return new IcpResult(R_total, t_total, currentRms, iter, converged, alignedCloud);
        }
    }
}
