"""add bounding box columns to person_detections

Revision ID: 20240321_000003
Revises: 20240321_000002
Create Date: 2024-03-21 00:00:03.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '20240321_000003'
down_revision = '20240321_000002'
branch_labels = None
depends_on = None


def upgrade():
    # Add bounding box columns
    op.add_column('person_detections', sa.Column('x1', sa.Float(), nullable=True))
    op.add_column('person_detections', sa.Column('y1', sa.Float(), nullable=True))
    op.add_column('person_detections', sa.Column('x2', sa.Float(), nullable=True))
    op.add_column('person_detections', sa.Column('y2', sa.Float(), nullable=True))
    
    # Add is_entry and is_exit columns
    op.add_column('person_detections', sa.Column('is_entry', sa.Boolean(), nullable=False, server_default='false'))
    op.add_column('person_detections', sa.Column('is_exit', sa.Boolean(), nullable=False, server_default='false'))


def downgrade():
    # Remove all added columns
    op.drop_column('person_detections', 'x1')
    op.drop_column('person_detections', 'y1')
    op.drop_column('person_detections', 'x2')
    op.drop_column('person_detections', 'y2')
    op.drop_column('person_detections', 'is_entry')
    op.drop_column('person_detections', 'is_exit') 