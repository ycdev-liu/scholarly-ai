"""反馈相关的 API 路由"""
from fastapi import APIRouter, Depends

from langsmith import Client as LangsmithClient
from schema import Feedback, FeedbackResponse
from service.utils import verify_bearer

router = APIRouter(prefix="/api/feedback", dependencies=[Depends(verify_bearer)])


@router.post("", operation_id="submit_feedback")
async def feedback(feedback: Feedback) -> FeedbackResponse:
    """提交反馈到 LangSmith"""
    client = LangsmithClient()
    kwargs = feedback.kwargs or {}
    client.create_feedback(
        run_id=feedback.run_id,
        key=feedback.key,
        score=feedback.score,
        **kwargs,
    )
    return FeedbackResponse()
