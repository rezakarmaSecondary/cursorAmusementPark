"""add detection_method to reports

Revision ID: 20240321_000004
Revises: 20240321_000003
Create Date: 2024-03-21 00:00:04.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '20240321_000004'
down_revision = '20240321_000003'
branch_labels = None
depends_on = None


def upgrade():
    # Add detection_method column to reports table
    op.add_column('reports', sa.Column('detection_method', sa.String(), nullable=False, server_default='camera_based'))


def downgrade():
    # Remove detection_method column
    op.drop_column('reports', 'detection_method') 