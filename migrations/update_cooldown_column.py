import os
import sys
from pathlib import Path

# Add the project root directory to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from sqlalchemy import text
from app.database import engine

def upgrade():
    # Rename the column from cooldown_minutes to cooldown_seconds
    with engine.connect() as connection:
        connection.execute(text("""
            ALTER TABLE devices 
            RENAME COLUMN cooldown_minutes TO cooldown_seconds;
        """))
        connection.commit()

def downgrade():
    # Revert the column name back to cooldown_minutes
    with engine.connect() as connection:
        connection.execute(text("""
            ALTER TABLE devices 
            RENAME COLUMN cooldown_seconds TO cooldown_minutes;
        """))
        connection.commit()

if __name__ == "__main__":
    upgrade()
    print("Successfully updated cooldown column from minutes to seconds") 