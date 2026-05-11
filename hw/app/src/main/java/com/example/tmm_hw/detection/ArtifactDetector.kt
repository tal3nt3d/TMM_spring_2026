package com.example.tmm_hw.detection

import org.opencv.core.Mat

data class DetectionResult(
    val mask: Mat,          // маска артефактов (белые пиксели = артефакт)
    val artifactCount: Int, // количество найденных артефактов
    val artifactRatio: Float // доля повреждённых пикселей (0..1)
)

interface ArtifactDetector {
    fun detect(frame: Mat): DetectionResult
    fun release()
}