from pydantic import BaseModel

class PatientData(BaseModel):
    age: float
    sex: float
    bmi: float
    bp: float
    s1: float
    s2: float
    s3: float
    s4: float
    s5: float
    s6: float

    class Config:
        json_schema_extra = {
            "example": {
                "age": 0.038076,
                "sex": 0.050680,
                "bmi": 0.061696,
                "bp": 0.021872,
                "s1": -0.044223,
                "s2": -0.034821,
                "s3": -0.043401,
                "s4": -0.002592,
                "s5": 0.019907,
                "s6": -0.017646
            }
        }
