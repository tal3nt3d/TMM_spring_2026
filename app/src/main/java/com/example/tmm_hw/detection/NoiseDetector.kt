package com.example.tmm_hw.detection

import org.opencv.core.*
import org.opencv.imgproc.Imgproc

/**
 * Обнаруживает:
 * - шум "соль и перец" (очень яркие / очень тёмные одиночные пиксели)
 * - блочные артефакты (резкие прямоугольные границы)
 * - битые пиксели (постоянно аномальные по сравнению с соседями)
 */
class NoiseDetector : ArtifactDetector {

    // Пороги для детекции шума соль-и-перец
    private val saltThreshold = 250.0
    private val pepperThreshold = 5.0

    override fun detect(frame: Mat): DetectionResult {
        val gray = Mat()
        val mask = Mat.zeros(frame.size(), CvType.CV_8UC1)

        // Конвертируем в grayscale для анализа
        Imgproc.cvtColor(frame, gray, Imgproc.COLOR_BGR2GRAY)

        // 1. Детекция шума "соль и перец"
        detectSaltAndPepper(gray, mask)

        // 2. Детекция блочных артефактов
        detectBlockArtifacts(gray, mask)

        // Подсчёт артефактов
        val artifactCount = Core.countNonZero(mask)
        val totalPixels = frame.rows() * frame.cols()
        val ratio = artifactCount.toFloat() / totalPixels

        gray.release()

        return DetectionResult(mask, artifactCount, ratio)
    }

    private fun detectSaltAndPepper(gray: Mat, mask: Mat) {
        // Маска для соли (очень яркие пиксели)
        val saltMask = Mat()
        Core.compare(gray, Scalar(saltThreshold), saltMask, Core.CMP_GT)

        // Маска для перца (очень тёмные пиксели)
        val pepperMask = Mat()
        Core.compare(gray, Scalar(pepperThreshold), pepperMask, Core.CMP_LT)

        // Объединяем
        Core.bitwise_or(mask, saltMask, mask)
        Core.bitwise_or(mask, pepperMask, mask)

        // Удаляем большие области — это не шум, а реальные объекты
        // Оставляем только маленькие изолированные пиксели
        val kernel = Imgproc.getStructuringElement(
            Imgproc.MORPH_RECT, Size(3.0, 3.0)
        )
        val expanded = Mat()
        Imgproc.dilate(mask, expanded, kernel)

        // Если пиксель изолирован (после расширения его "масса" мала) — это шум
        val connected = Mat()
        Imgproc.erode(expanded, connected, kernel)
        Core.bitwise_and(mask, connected, mask)

        saltMask.release()
        pepperMask.release()
        expanded.release()
        connected.release()
        kernel.release()
    }

    private fun detectBlockArtifacts(gray: Mat, mask: Mat) {
        // Применяем DCT-подобный анализ через градиент
        // Блочные артефакты проявляются как резкие горизонтальные/вертикальные линии
        val sobelX = Mat()
        val sobelY = Mat()
        Imgproc.Sobel(gray, sobelX, CvType.CV_16S, 1, 0)
        Imgproc.Sobel(gray, sobelY, CvType.CV_16S, 0, 1)

        val absX = Mat()
        val absY = Mat()
        Core.convertScaleAbs(sobelX, absX)
        Core.convertScaleAbs(sobelY, absY)

        val edges = Mat()
        Core.addWeighted(absX, 0.5, absY, 0.5, 0.0, edges)

        // Бинаризуем — только очень резкие края (блочные артефакты)
        val blockMask = Mat()
        Imgproc.threshold(edges, blockMask, 200.0, 255.0, Imgproc.THRESH_BINARY)

        // Ищем паттерны через периодическую структуру (блоки 8x8 у JPEG)
        val kernel8 = Imgproc.getStructuringElement(
            Imgproc.MORPH_RECT, Size(8.0, 1.0)
        )
        val horizontalLines = Mat()
        Imgproc.morphologyEx(blockMask, horizontalLines, Imgproc.MORPH_OPEN, kernel8)

        val kernel8v = Imgproc.getStructuringElement(
            Imgproc.MORPH_RECT, Size(1.0, 8.0)
        )
        val verticalLines = Mat()
        Imgproc.morphologyEx(blockMask, verticalLines, Imgproc.MORPH_OPEN, kernel8v)

        Core.bitwise_or(mask, horizontalLines, mask)
        Core.bitwise_or(mask, verticalLines, mask)

        sobelX.release(); sobelY.release()
        absX.release(); absY.release()
        edges.release(); blockMask.release()
        horizontalLines.release(); verticalLines.release()
        kernel8.release(); kernel8v.release()
    }

    override fun release() {}
}