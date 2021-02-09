import React, { useRef, useEffect, useState } from 'react';


const Canvas = props => {
  const canvasRef = useRef(null)
  const contextRef = useRef(null)
  const [isDrawing, setIsDrawing] = useState(false)
  const [curr_x, set_curr_x] = useState(false)
  const [curr_y, set_curr_y] = useState(false)

  useEffect(() => {
    const canvas = canvasRef.current;
    var innerWidth = window.innerWidth*.997;
    var innerHeight = window.innerHeight*.997;
    canvas.width = innerWidth * 2;
    canvas.height = innerHeight * 2;
    canvas.style.width = `${innerWidth}px`;
    canvas.style.height = `${innerHeight}px`;

    const context = canvas.getContext("2d")
    context.scale(2,2)
    context.lineCap = "round"
    context.strokeStyle = "black"
    context.lineWidth = 5
    contextRef.current = context;
  }, [])

  const startDrawing = ({nativeEvent}) => {
    const {offsetX, offsetY} = nativeEvent;
    set_curr_x(offsetX);
    set_curr_y(offsetY);
    setIsDrawing(true)
  }

  const draw = ({nativeEvent}) => {
    //console.log(nativeEvent.pressure);

    const {offsetX, offsetY, pressure} = nativeEvent;

    if (pressure <= 0){
      return
    }
    if(!isDrawing){
      return
    }
    contextRef.current.beginPath()
    contextRef.current.moveTo(curr_x, curr_y)
    contextRef.current.lineTo(offsetX, offsetY)
    contextRef.current.stroke()

    set_curr_x(offsetX);
    set_curr_y(offsetY);
  }

  return (
    <div>
    <canvas style={{"touch-action": "none"}}
      onPointerDown={startDrawing}
      onPointerMove={draw}
      ref={canvasRef}
    />

    </div>

  );
}

export default Canvas
