package com.example.tmm_hw.removal

import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc

class MedianRemover : ArtifactRemover {

    override fun remove(frame: Mat, mask: Mat): Mat {
        val result = Mat()
        // Медианный фильтр 5x5 — эффективно убирает шум соль-и-перец
        Imgproc.medianBlur(frame, result, 5)

        // Применяем фильтр только к пикселям с артефактами
        val output = frame.clone()
        result.copyTo(output, mask)

        result.release()
        return output
    }

    override fun release() {}
}