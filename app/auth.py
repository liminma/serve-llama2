from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from api_keys import API_KEYS


api_key_header = APIKeyHeader(name="Authorization", auto_error=False)


def retrieve_api_key(api_key: str = Security(api_key_header)) -> None:
    """Validate the API key from HTTP header.

    Args:
        api_key (str, optional): API key starting with 'Bearer '.

    Raises:
        HTTPException: HTTP status code 401.
    """
    if api_key is None or not api_key[len('Bearer '):] in API_KEYS:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="A valid API Key is required.")
