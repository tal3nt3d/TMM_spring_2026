package com.example.tmm_hw.removal

import org.opencv.core.Mat

enum class RemovalMode {
    MEDIAN,     // медианный фильтр — лучший для шума соль-и-перец
    BILATERAL,  // bilateral filter — сохраняет края
    INPAINT     // inpainting — заполняет по соседям
}

interface ArtifactRemover {
    fun remove(frame: Mat, mask: Mat): Mat
    fun release()
}