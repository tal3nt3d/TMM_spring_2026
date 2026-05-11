package com.example.tmm_hw.pipeline

import com.example.tmm_hw.detection.DetectionResult
import com.example.tmm_hw.detection.NoiseDetector
import com.example.tmm_hw.removal.*
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import org.opencv.photo.Photo

data class PipelineResult(
    val processedFrame: Mat,
    val detectionResult: DetectionResult,
    val processingTimeMs: Long
)

class ProcessingPipeline {

    private val detector = NoiseDetector()

    var currentMode: RemovalMode = RemovalMode.MEDIAN
    var processingEnabled: Boolean = true
    var testNoiseEnabled: Boolean = false

    fun process(frame: Mat): PipelineResult {
        val startTime = System.currentTimeMillis()

        val workFrame = frame.clone()

        if (testNoiseEnabled) {
            addTestNoise(workFrame)
        }

        if (!processingEnabled) {
            val emptyMask = Mat.zeros(workFrame.size(), CvType.CV_8UC1)
            return PipelineResult(workFrame, DetectionResult(emptyMask, 0, 0f), 0L)
        }

        val processedFrame = when (currentMode) {
            RemovalMode.MEDIAN -> {
                val result = Mat()
                Imgproc.medianBlur(workFrame, result, 5)
                workFrame.release()
                result
            }
            RemovalMode.BILATERAL -> {
                val result = Mat()
                Imgproc.bilateralFilter(workFrame, result, 9, 75.0, 75.0)
                workFrame.release()
                result
            }
            RemovalMode.INPAINT -> {
                val mask = buildInpaintMask(workFrame)
                val result = Mat()
                // Увеличиваем радиус до 10 для лучшего восстановления царапин
                Photo.inpaint(workFrame, mask, result, 10.0, Photo.INPAINT_NS)
                mask.release()
                workFrame.release()
                result
            }
        }

        val detection = detector.detect(processedFrame)
        val elapsed = System.currentTimeMillis() - startTime

        return PipelineResult(processedFrame, detection, elapsed)
    }

    private fun buildInpaintMask(frame: Mat): Mat {
        val gray = Mat()
        Imgproc.cvtColor(frame, gray, Imgproc.COLOR_BGR2GRAY)

        // 1. Абсолютные пороги — снижаем чтобы ловить размытые артефакты
        val saltMask = Mat()
        val pepperMask = Mat()
        Core.compare(gray, Scalar(220.0), saltMask, Core.CMP_GT)  // было 245
        Core.compare(gray, Scalar(35.0), pepperMask, Core.CMP_LT) // было 10
        val absoluteMask = Mat()
        Core.bitwise_or(saltMask, pepperMask, absoluteMask)

        // 2. Детекция линий через Sobel — царапины это резкие перепады яркости
        val sobelX = Mat()
        val sobelY = Mat()
        Imgproc.Sobel(gray, sobelX, CvType.CV_16S, 1, 0, 3)
        Imgproc.Sobel(gray, sobelY, CvType.CV_16S, 0, 1, 3)
        val absX = Mat()
        val absY = Mat()
        Core.convertScaleAbs(sobelX, absX)
        Core.convertScaleAbs(sobelY, absY)
        val edges = Mat()
        Core.addWeighted(absX, 0.5, absY, 0.5, 0.0, edges)

        // Порог для резких краёв — царапины дают очень сильный градиент
        val edgeMask = Mat()
        Imgproc.threshold(edges, edgeMask, 80.0, 255.0, Imgproc.THRESH_BINARY)

        // 3. Детекция через локальное отклонение от среднего
        val blurred = Mat()
        Imgproc.GaussianBlur(gray, blurred, Size(7.0, 7.0), 0.0)
        val diff = Mat()
        Core.absdiff(gray, blurred, diff)
        val localMask = Mat()
        Imgproc.threshold(diff, localMask, 30.0, 255.0, Imgproc.THRESH_BINARY)

        // 4. Объединяем все три маски
        val combinedMask = Mat()
        Core.bitwise_or(absoluteMask, edgeMask, combinedMask)
        Core.bitwise_or(combinedMask, localMask, combinedMask)

        // 5. Морфология — закрываем разрывы в царапинах
        val kernelClose = Imgproc.getStructuringElement(
            Imgproc.MORPH_RECT, Size(5.0, 1.0) // горизонтальное закрытие
        )
        val kernelClose2 = Imgproc.getStructuringElement(
            Imgproc.MORPH_RECT, Size(1.0, 5.0) // вертикальное закрытие
        )
        Imgproc.morphologyEx(combinedMask, combinedMask, Imgproc.MORPH_CLOSE, kernelClose)
        Imgproc.morphologyEx(combinedMask, combinedMask, Imgproc.MORPH_CLOSE, kernelClose2)

        // Убираем мелкий мусор
        val kernelOpen = Imgproc.getStructuringElement(
            Imgproc.MORPH_ELLIPSE, Size(2.0, 2.0)
        )
        Imgproc.morphologyEx(combinedMask, combinedMask, Imgproc.MORPH_OPEN, kernelOpen)

        // Расширяем маску — inpaint должен захватить края
        val kernelDilate = Imgproc.getStructuringElement(
            Imgproc.MORPH_ELLIPSE, Size(4.0, 4.0)
        )
        Imgproc.dilate(combinedMask, combinedMask, kernelDilate)

        gray.release()
        saltMask.release()
        pepperMask.release()
        absoluteMask.release()
        sobelX.release(); sobelY.release()
        absX.release(); absY.release()
        edges.release(); edgeMask.release()
        blurred.release(); diff.release(); localMask.release()
        kernelClose.release(); kernelClose2.release()
        kernelOpen.release(); kernelDilate.release()

        return combinedMask
    }

    private fun addTestNoise(frame: Mat) {
        val mask = Mat(frame.size(), CvType.CV_8UC1)
        Core.randu(mask, 0.0, 255.0)

        val saltMask = Mat()
        Core.compare(mask, Scalar(240.0), saltMask, Core.CMP_GT)
        frame.setTo(Scalar(255.0, 255.0, 255.0), saltMask)

        val pepperMask = Mat()
        Core.compare(mask, Scalar(15.0), pepperMask, Core.CMP_LT)
        frame.setTo(Scalar(0.0, 0.0, 0.0), pepperMask)

        mask.release()
        saltMask.release()
        pepperMask.release()
    }

    fun release() {
        detector.release()
    }
}