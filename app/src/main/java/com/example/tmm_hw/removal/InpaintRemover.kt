package com.example.tmm_hw.removal
import org.opencv.core.Mat
import org.opencv.photo.Photo

class InpaintRemover : ArtifactRemover {

    override fun remove(frame: Mat, mask: Mat): Mat {
        val result = Mat()
        // INPAINT_TELEA — быстрый алгоритм заполнения на основе соседних пикселей
        Photo.inpaint(frame, mask, result, 3.0, Photo.INPAINT_TELEA)
        return result
    }

    override fun release() {}
}