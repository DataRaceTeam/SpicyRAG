from interface import (AddHeaderTransformChunk,
                       RecursiveCharacterTextSplitterChunker)


class TestRecursiveCharacterTextSplitterChunker:
    def test_chunk(self):
        chunker = RecursiveCharacterTextSplitterChunker(chunk_size=4, chunk_overlap=2)

        chunks = chunker.chunk("abcdefgh")
        assert chunks == ["abcd", "cdef", "efgh"]


class TestAddHeaderTransformChunk:
    def test_chunk(self, text_hmao_npa_df):
        text = text_hmao_npa_df["document"].iloc[0]
        chunks = ["a"]

        actual = AddHeaderTransformChunk().transform(chunks, text)
        assert actual == [
            'ПОСТАНОВЛЕНИЕ ГУБЕРНАТОРА ХАНТЫ-МАНСИЙСКОГО АВТОНОМНОГО ОКРУГА-ЮГРЫ от 28.12.2017 № 139.  О ВНЕСЕНИИ ИЗМЕНЕНИЙ В ПРИЛОЖЕНИЕ К ПОСТАНОВЛЕНИЮ ГУБЕРНАТОРА ХАНТЫ-МАНСИЙСКОГО АВТОНОМНОГО ОКРУГА – ЮГРЫ ОТ 30 ДЕКАБРЯ 2012 ГОДА N 176 "ОБ ИНСТРУКЦИИ ПО ДЕЛОПРОИЗВОДСТВУ В ГОСУДАРСТВЕННЫХ ОРГАНАХ ХАНТЫ-МАНСИЙСКОГО АВТОНОМНОГО ОКРУГА - ЮГРЫ И ИСПОЛНИТЕЛЬНЫХ ОРГАНАХ ГОСУДАРСТВЕННОЙ ВЛАСТИ ХАНТЫ-МАНСИЙСКОГО АВТОНОМНОГО ОКРУГА - ЮГРЫ"a'
        ]
