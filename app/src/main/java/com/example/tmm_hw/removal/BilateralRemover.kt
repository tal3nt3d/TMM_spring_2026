package com.example.tmm_hw.removal

import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc

class BilateralRemover : ArtifactRemover {

    override fun remove(frame: Mat, mask: Mat): Mat {
        val filtered = Mat()
        // Bilateral filter: сохраняет края, размывает однородные зоны
        // d=9, sigmaColor=75, sigmaSpace=75
        Imgproc.bilateralFilter(frame, filtered, 9, 75.0, 75.0)

        val output = frame.clone()
        filtered.copyTo(output, mask)

        filtered.release()
        return output
    }

    override fun release() {}
}