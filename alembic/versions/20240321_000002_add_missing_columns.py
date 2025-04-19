"""add missing columns to person_detections

Revision ID: 20240321_000002
Revises: 20240321_000001
Create Date: 2024-03-21 00:00:02.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '20240321_000002'
down_revision = '20240321_000001'
branch_labels = None
depends_on = None


def upgrade():
    # Add detection_time column
    op.add_column('person_detections', sa.Column('detection_time', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')))


def downgrade():
    # Remove detection_time column
    op.drop_column('person_detections', 'detection_time') 