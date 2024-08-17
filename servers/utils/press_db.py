from sqlmodel import Field, SQLModel, create_engine, Session, select


class Clothes(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(sa_column_kwargs={"unique": True})
    url: str


sqlite_file_name = "clothes.db"
sqlite_url = f"sqlite:///db/{sqlite_file_name}"

engine = create_engine(sqlite_url, echo=True)


def create_db_and_tables():
    SQLModel.metadata.create_all(engine)


def drop_db_and_tables():
    SQLModel.metadata.drop_all(engine)

def add_clothes(clothes: Clothes): 
    with Session(engine) as session:
        session.add(clothes)
        session.commit()

def get_clothes(name: str):
    clothes = None
    with Session(engine) as session:
        statement = select(Clothes.url).where(Clothes.name == name)
        clothes = session.exec(statement).first()
    return clothes

if __name__ == "__main__":
    create_db_and_tables()