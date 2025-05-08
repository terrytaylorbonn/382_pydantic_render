# docs/examples/pydantic_ai_examples/flight_booking_api.py

import datetime
from dataclasses import dataclass
from typing import Literal, Union

import logfire
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.messages import ModelMessage
from pydantic_ai.usage import Usage, UsageLimits

# logfire.configure(service_name="flight_booking_app")
# logfire.instrument_httpx()

app = FastAPI()

# === Models ===

class FlightDetails(BaseModel):
    flight_number: str
    price: int
    origin: str
    destination: str
    date: datetime.date

class NoFlightFound(BaseModel):
    pass

class SeatPreference(BaseModel):
    row: int = Field(ge=1, le=30)
    seat: Literal['A', 'B', 'C', 'D', 'E', 'F']

class Failed(BaseModel):
    pass

@dataclass
class Deps:
    web_page_text: str
    req_origin: str
    req_destination: str
    req_date: datetime.date

# === Agents ===

search_agent = Agent[Deps, FlightDetails | NoFlightFound](
    'openai:gpt-4o',
    output_type=FlightDetails | NoFlightFound,  # type: ignore
    retries=4,
    system_prompt='Your job is to find the cheapest flight for the user on the given date.',
    instrument=True,
)

extraction_agent = Agent(
    'openai:gpt-4o',
    output_type=list[FlightDetails],
    system_prompt='Extract all the flight details from the given text.',
)

seat_preference_agent = Agent[None, SeatPreference | Failed](
    'openai:gpt-4o',
    output_type=SeatPreference | Failed,  # type: ignore
    system_prompt=(
        "Extract the user's seat preference. "
        'Seats A and F are window seats. '
        'Row 1 is the front row and has extra leg room. '
        'Rows 14, and 20 also have extra leg room. '
    ),
)

@search_agent.tool
async def extract_flights(ctx: RunContext[Deps]) -> list[FlightDetails]:
    result = await extraction_agent.run(ctx.deps.web_page_text, usage=ctx.usage)
    logfire.info('found {flight_count} flights', flight_count=len(result.output))
    return result.output

@search_agent.output_validator
async def validate_output(ctx: RunContext[Deps], output: FlightDetails | NoFlightFound):
    if isinstance(output, NoFlightFound):
        return output

    errors = []
    if output.origin != ctx.deps.req_origin:
        errors.append(f'Origin mismatch: {output.origin}')
    if output.destination != ctx.deps.req_destination:
        errors.append(f'Destination mismatch: {output.destination}')
    if output.date != ctx.deps.req_date:
        errors.append(f'Date mismatch: {output.date}')

    if errors:
        raise ModelRetry('\n'.join(errors))
    return output

# === Fake web page ===

flights_web_page = """ (same large text block you have) """

usage_limits = UsageLimits(request_limit=15)

# === FastAPI Endpoints ===

class SearchRequest(BaseModel):
    origin: str
    destination: str
    date: datetime.date

class SeatRequest(BaseModel):
    seat_text: str

@app.post("/search_flight")
async def search_flight(req: SearchRequest):
    deps = Deps(
        web_page_text=flights_web_page,
        req_origin=req.origin,
        req_destination=req.destination,
        req_date=req.date,
    )
    usage = Usage()
    result = await search_agent.run(
        f'Find me a flight from {req.origin} to {req.destination} on {req.date}',
        deps=deps,
        usage=usage,
        usage_limits=usage_limits,
    )
    if isinstance(result.output, NoFlightFound):
        raise HTTPException(status_code=404, detail="No flight found")
    return result.output

@app.post("/select_seat")
async def select_seat(req: SeatRequest):
    usage = Usage()
    result = await seat_preference_agent.run(
        req.seat_text,
        usage=usage,
        usage_limits=usage_limits,
    )
    if isinstance(result.output, Failed):
        raise HTTPException(status_code=400, detail="Could not parse seat preference")
    return result.output

@app.post("/buy_ticket")
async def buy_ticket(flight: FlightDetails, seat: SeatPreference):
    logfire.info('Purchasing ticket', flight=flight, seat=seat)
    return {"message": f"Purchased {flight.flight_number} seat {seat.row}{seat.seat}"}
