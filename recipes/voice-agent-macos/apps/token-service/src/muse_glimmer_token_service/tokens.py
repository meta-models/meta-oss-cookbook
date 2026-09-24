from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import timedelta
from typing import Final

from livekit import api

from .config import Settings

AGENT_NAME: Final = "assistant"


@dataclass(frozen=True, slots=True)
class ConnectionDetails:
    server_url: str
    participant_token: str
    room_name: str
    participant_identity: str


def issue_connection(settings: Settings) -> ConnectionDetails:
    room_name = f"r_{uuid.uuid4().hex}"
    participant_identity = f"p_{uuid.uuid4().hex}"
    grants = api.VideoGrants(
        room_join=True,
        room=room_name,
        can_publish=True,
        can_subscribe=True,
        can_publish_data=False,
        can_publish_sources=["microphone"],
    )
    room_config = api.RoomConfiguration(
        agents=[api.RoomAgentDispatch(agent_name=AGENT_NAME)],
    )
    participant_token = (
        api.AccessToken(
            settings.livekit_api_key.get_secret_value(),
            settings.livekit_api_secret.get_secret_value(),
        )
        .with_identity(participant_identity)
        .with_name("Local voice participant")
        .with_ttl(timedelta(seconds=settings.token_ttl_seconds))
        .with_grants(grants)
        .with_room_config(room_config)
        .to_jwt()
    )

    return ConnectionDetails(
        server_url=settings.livekit_url,
        participant_token=participant_token,
        room_name=room_name,
        participant_identity=participant_identity,
    )
