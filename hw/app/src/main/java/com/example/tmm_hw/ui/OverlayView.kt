package com.example.tmm_hw.ui

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc

class OverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : View(context, attrs) {

    private var overlayBitmap: Bitmap? = null
    private val maskPaint = Paint().apply {
        color = Color.argb(120, 255, 0, 0) // полупрозрачный красный
        style = Paint.Style.FILL
    }

    fun updateFrame(processedMat: Mat, maskMat: Mat) {
        val rgbMat = Mat()
        Imgproc.cvtColor(processedMat, rgbMat, Imgproc.COLOR_BGR2RGBA)

        val bitmap = Bitmap.createBitmap(
            rgbMat.cols(), rgbMat.rows(), Bitmap.Config.ARGB_8888
        )
        Utils.matToBitmap(rgbMat, bitmap)
        rgbMat.release()

        overlayBitmap = bitmap
        postInvalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        overlayBitmap?.let { bmp ->
            // Вписываем с сохранением пропорций (centerCrop)
            val viewRatio = width.toFloat() / height.toFloat()
            val bmpRatio = bmp.width.toFloat() / bmp.height.toFloat()

            val srcRect: Rect
            if (bmpRatio > viewRatio) {
                val crop = ((bmp.width - bmp.height * viewRatio) / 2).toInt()
                srcRect = Rect(crop, 0, bmp.width - crop, bmp.height)
            } else {
                val crop = ((bmp.height - bmp.width / viewRatio) / 2).toInt()
                srcRect = Rect(0, crop, bmp.width, bmp.height - crop)
            }

            val dstRect = Rect(0, 0, width, height)
            canvas.drawBitmap(bmp, srcRect, dstRect, null)
        }
    }
}