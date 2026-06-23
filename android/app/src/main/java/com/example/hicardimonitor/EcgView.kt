package com.example.hicardimonitor

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View
import kotlin.math.abs
import kotlin.math.max

class EcgView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : View(context, attrs) {

    private val values = ArrayDeque<Float>()
    private val maxSamples = 1250

    /** 샘플링 주파수 (PQRST 윈도우 계산용). MainActivity에서 JSON 로드 시 설정. */
    var fs: Float = 250f

    // 최근 1비트의 PQRST 마커: (끝에서부터의 거리 offsetFromEnd, 라벨, 값)
    private data class Marker(val offsetFromEnd: Int, val label: String, val value: Float)
    private var markers: List<Marker> = emptyList()

    private val gridPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.rgb(25, 48, 80)
        strokeWidth = 1f
    }
    private val centerPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.rgb(42, 96, 144)
        strokeWidth = 2f
    }
    private val linePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.rgb(0, 229, 255)
        strokeWidth = 4f
        style = Paint.Style.STROKE
    }
    private val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.rgb(90, 122, 154)
        textSize = 28f
        textAlign = Paint.Align.CENTER
    }
    private val markerDotPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
    }
    private val markerTextPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        textSize = 26f
        textAlign = Paint.Align.CENTER
        isFakeBoldText = true
    }

    private val pqrstColors = mapOf(
        "P" to Color.rgb(0, 207, 255),
        "Q" to Color.rgb(255, 152, 0),
        "R" to Color.rgb(255, 51, 51),
        "S" to Color.rgb(255, 152, 0),
        "T" to Color.rgb(0, 255, 136)
    )

    fun addWaveform(waveform: List<Float>) {
        for (v in waveform) {
            values.addLast(v)
            while (values.size > maxSamples) values.removeFirst()
        }
        markers = detectPqrst(waveform)
        invalidate()
    }

    fun clear() {
        values.clear()
        markers = emptyList()
        invalidate()
    }

    /**
     * 한 비트(waveform) 안에서 P/Q/R/S/T 위치를 추정한다.
     * (파이썬 원본 _detect_pqrst 와 동일한 규칙)
     */
    private fun detectPqrst(waveform: List<Float>): List<Marker> {
        val n = waveform.size
        if (n < 10) return emptyList()

        fun maxIdx(from: Int, to: Int): Int {
            var bi = from; var bv = waveform[from]
            for (i in from until to) if (waveform[i] > bv) { bv = waveform[i]; bi = i }
            return bi
        }
        fun minIdx(from: Int, to: Int): Int {
            var bi = from; var bv = waveform[from]
            for (i in from until to) if (waveform[i] < bv) { bv = waveform[i]; bi = i }
            return bi
        }

        // R: 중앙 ±15% 범위 최댓값
        val rCenter = n / 2
        val search = (fs * 0.15f).toInt()
        val rStart = max(0, rCenter - search)
        val rEnd = minOf(n, rCenter + search)
        val rIdx = maxIdx(rStart, rEnd)

        // Q: R 직전 20ms 최솟값
        val qWin = max(1, (fs * 0.02f).toInt())
        val qIdx = if (rIdx - qWin >= 0) minIdx(max(0, rIdx - qWin), rIdx) else max(0, rIdx - 2)

        // S: R 직후 20ms 최솟값
        val sWin = max(1, (fs * 0.02f).toInt())
        val sIdx = if (rIdx + 1 < minOf(n, rIdx + sWin + 1)) minIdx(rIdx + 1, minOf(n, rIdx + sWin + 1))
                   else minOf(n - 1, rIdx + 2)

        // P: R 이전 80~200ms 최댓값
        val pStart = max(0, rIdx - (fs * 0.20f).toInt())
        val pEnd = max(0, rIdx - (fs * 0.08f).toInt())
        val pIdx = if (pEnd > pStart) maxIdx(pStart, pEnd) else max(0, rIdx - (fs * 0.14f).toInt())

        // T: R 이후 100~350ms 최댓값
        val tStart = minOf(n, rIdx + (fs * 0.10f).toInt())
        val tEnd = minOf(n, rIdx + (fs * 0.35f).toInt())
        val tIdx = if (tEnd > tStart) maxIdx(tStart, tEnd) else minOf(n - 1, rIdx + (fs * 0.20f).toInt())

        // offsetFromEnd = 비트 끝에서부터의 거리 (버퍼가 앞에서 잘려도 안정적)
        return listOf(
            Marker(n - pIdx, "P", waveform[pIdx]),
            Marker(n - qIdx, "Q", waveform[qIdx]),
            Marker(n - rIdx, "R", waveform[rIdx]),
            Marker(n - sIdx, "S", waveform[sIdx]),
            Marker(n - tIdx, "T", waveform[tIdx])
        )
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        canvas.drawColor(Color.rgb(10, 22, 40))

        val w = width.toFloat()
        val h = height.toFloat()
        val pad = 24f
        val midY = h / 2f

        for (x in 0..width step 40) canvas.drawLine(x.toFloat(), 0f, x.toFloat(), h, gridPaint)
        for (y in 0..height step 40) canvas.drawLine(0f, y.toFloat(), w, y.toFloat(), gridPaint)
        canvas.drawLine(0f, midY, w, midY, centerPaint)

        if (values.size < 2) {
            canvas.drawText("JSON 파일을 열고 재생하세요", w / 2f, h / 2f, textPaint)
            return
        }

        val data = values.toList()
        val maxAmp = max(0.5f, data.maxOf { abs(it) } * 1.2f)
        val xStep = (w - pad * 2) / (data.size - 1)
        val yScale = (h * 0.42f) / maxAmp

        var prevX = pad
        var prevY = midY - data[0] * yScale
        for (i in 1 until data.size) {
            val x = pad + i * xStep
            val y = midY - data[i] * yScale
            canvas.drawLine(prevX, prevY.coerceIn(0f, h), x, y.coerceIn(0f, h), linePaint)
            prevX = x
            prevY = y
        }

        // ── PQRST 마커 (최근 1비트) ──
        for (mk in markers) {
            val idx = data.size - mk.offsetFromEnd
            if (idx < 0 || idx >= data.size) continue
            val mx = pad + idx * xStep
            val my = (midY - data[idx] * yScale).coerceIn(0f, h)
            val col = pqrstColors[mk.label] ?: Color.WHITE
            markerDotPaint.color = col
            canvas.drawCircle(mx, my, 7f, markerDotPaint)
            markerTextPaint.color = col
            val dy = if (mk.label == "R" || mk.label == "P" || mk.label == "T") -18f else 30f
            canvas.drawText(mk.label, mx, my + dy, markerTextPaint)
        }
    }
}
