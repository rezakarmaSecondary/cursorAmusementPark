"""add device_id to person_detections

Revision ID: 20240321_000001
Revises: initial
Create Date: 2024-03-21 00:00:01.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '20240321_000001'
down_revision = 'initial'
branch_labels = None
depends_on = None


def upgrade():
    # Add device_id column to person_detections table
    op.add_column('person_detections', sa.Column('device_id', sa.Integer(), nullable=True))
    
    # Add foreign key constraint
    op.create_foreign_key(
        'fk_person_detections_device_id',
        'person_detections', 'devices',
        ['device_id'], ['id'],
        ondelete='CASCADE'
    )
    
    # Update existing records to set device_id based on camera's device_id
    op.execute("""
        UPDATE person_detections pd
        SET device_id = c.device_id
        FROM cameras c
        WHERE pd.camera_id = c.id
    """)
    
    # Make device_id non-nullable after updating existing records
    op.alter_column('person_detections', 'device_id', nullable=False)


def downgrade():
    # Remove foreign key constraint
    op.drop_constraint('fk_person_detections_device_id', 'person_detections', type_='foreignkey')
    
    # Remove device_id column
    op.drop_column('person_detections', 'device_id') 