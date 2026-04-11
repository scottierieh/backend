from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import numpy as np

router = APIRouter()

class GradientDescentRequest(BaseModel):
    learning_rate: float = Field(default=0.1)
    start_x: float = Field(default=4.0)
    start_y: float = Field(default=4.0)
    num_steps: int = Field(default=50)

@router.post("/gradient-descent")
def gradient_descent(req: GradientDescentRequest):
    try:
        lr = req.learning_rate
        x, y = req.start_x, req.start_y
        num_steps = req.num_steps
        
        path = []
        
        for _ in range(num_steps):
            z = x**2 + y**2  # f(x,y) = x² + y²
            path.append([float(x), float(y), float(z)])
            
            # Gradients: df/dx = 2x, df/dy = 2y
            grad_x = 2 * x
            grad_y = 2 * y
            
            # Update
            x = x - lr * grad_x
            y = y - lr * grad_y
        
        # Final point
        z = x**2 + y**2
        path.append([float(x), float(y), float(z)])
        
        return {
            'path': path,
            'function_expression': 'f(x, y) = x² + y²',
            'parameters': {
                'learning_rate': lr,
                'start_x': req.start_x,
                'start_y': req.start_y,
                'num_steps': num_steps
            },
            'final_position': {'x': float(x), 'y': float(y), 'z': float(z)}
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
