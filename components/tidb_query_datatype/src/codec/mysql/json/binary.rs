// Copyright 2017 TiKV Project Authors. Licensed under Apache-2.0.

use std::convert::TryInto;

use codec::number::NumberCodec;

use super::{ERR_CONVERT_FAILED, JsonRef, JsonType, constants::*};
use crate::codec::{Error, Result, convert::ToStringValue, mysql::json::path_expr::ArrayIndex};

impl<'a> JsonRef<'a> {
    /// Bounds-checked slice of the binary payload. Corrupt/truncated JSON must
    /// return `Err` (coprocessor error), not panic — untrusted storage/network
    /// bytes can poison offsets.
    #[inline]
    fn try_slice(&self, start: usize, end: usize) -> Result<&'a [u8]> {
        self.value().get(start..end).ok_or_else(|| {
            Error::CorruptedData(format!(
                "JSON binary out of bounds: [{}, {}) of len {}",
                start,
                end,
                self.value().len()
            ))
        })
    }

    #[inline]
    fn try_decode_u32_le_at(&self, off: usize) -> Result<u32> {
        let end = off.checked_add(U32_LEN).ok_or_else(|| {
            Error::CorruptedData(format!("JSON binary u32 offset overflow at {}", off))
        })?;
        Ok(NumberCodec::decode_u32_le(self.try_slice(off, end)?))
    }

    #[inline]
    fn try_decode_u16_le_at(&self, off: usize) -> Result<u16> {
        let end = off.checked_add(U16_LEN).ok_or_else(|| {
            Error::CorruptedData(format!("JSON binary u16 offset overflow at {}", off))
        })?;
        Ok(NumberCodec::decode_u16_le(self.try_slice(off, end)?))
    }

    /// Gets the index from the ArrayIndex
    ///
    /// If the idx is greater than the count and is from right, it will return
    /// `None`
    ///
    /// See `jsonPathArrayIndex.getIndexFromStart()` in TiDB
    /// `types/json_path_expr.go`
    pub fn array_get_index(&self, idx: ArrayIndex) -> Option<usize> {
        match idx {
            ArrayIndex::Left(idx) => Some(idx as usize),
            ArrayIndex::Right(idx) => {
                if self.get_elem_count() < 1 + (idx as usize) {
                    None
                } else {
                    Some(self.get_elem_count() - 1 - (idx as usize))
                }
            }
        }
    }

    /// Gets the ith element in JsonRef
    ///
    /// See `arrayGetElem()` in TiDB `json/binary.go`
    pub fn array_get_elem(&self, idx: usize) -> Result<JsonRef<'a>> {
        let off = HEADER_LEN
            .checked_add(idx.checked_mul(VALUE_ENTRY_LEN).ok_or_else(|| {
                Error::CorruptedData("JSON array index overflow".into())
            })?)
            .ok_or_else(|| Error::CorruptedData("JSON array entry offset overflow".into()))?;
        self.val_entry_get(off)
    }

    /// Return the `i`th key in current Object json
    ///
    /// See `objectGetKey()` in TiDB `types/json_binary.go`
    ///
    /// # Errors
    ///
    /// Returns `Error::CorruptedData` if key-entry offsets are out of bounds
    /// (truncated or poison binary). Does not panic on untrusted input.
    pub fn object_get_key(&self, i: usize) -> Result<&'a [u8]> {
        let key_off_start = HEADER_LEN
            .checked_add(i.checked_mul(KEY_ENTRY_LEN).ok_or_else(|| {
                Error::CorruptedData("JSON object key index overflow".into())
            })?)
            .ok_or_else(|| Error::CorruptedData("JSON object key-entry offset overflow".into()))?;
        let key_off = self.try_decode_u32_le_at(key_off_start)? as usize;
        let key_len = self.try_decode_u16_le_at(key_off_start + KEY_OFFSET_LEN)? as usize;
        let key_end = key_off.checked_add(key_len).ok_or_else(|| {
            Error::CorruptedData(format!(
                "JSON object key range overflow: off={} len={}",
                key_off, key_len
            ))
        })?;
        self.try_slice(key_off, key_end)
    }

    /// Returns the JsonRef of `i`th value in current Object json
    ///
    /// See `objectGetVal()` in TiDB `types/json_binary.go`
    pub fn object_get_val(&self, i: usize) -> Result<JsonRef<'a>> {
        let ele_count = self.get_elem_count();
        let keys_bytes = ele_count.checked_mul(KEY_ENTRY_LEN).ok_or_else(|| {
            Error::CorruptedData("JSON object key-entries size overflow".into())
        })?;
        let val_entry_off = HEADER_LEN
            .checked_add(keys_bytes)
            .and_then(|b| b.checked_add(i.checked_mul(VALUE_ENTRY_LEN)?))
            .ok_or_else(|| Error::CorruptedData("JSON object value-entry offset overflow".into()))?;
        self.val_entry_get(val_entry_off)
    }

    /// Searches the value index by the give `key` in Object.
    ///
    /// See `objectSearchKey()` in TiDB `json/binary_function.go`
    ///
    /// # Errors
    ///
    /// Propagates `object_get_key` corruption errors instead of panicking.
    pub fn object_search_key(&self, key: &[u8]) -> Result<Option<usize>> {
        let len = self.get_elem_count();
        let mut j = len;
        let mut i = 0;
        while i < j {
            let mid = (i + j) >> 1;
            if self.object_get_key(mid)? < key {
                i = mid + 1;
            } else {
                j = mid;
            }
        }
        if i < len && self.object_get_key(i)? == key {
            return Ok(Some(i));
        }
        Ok(None)
    }

    /// Gets the value (JsonRef) by the given offset of the value entry
    ///
    /// See `arrayGetElem()` in TiDB `json/binary.go`
    pub fn val_entry_get(&self, val_entry_off: usize) -> Result<JsonRef<'a>> {
        let type_bytes = self.try_slice(val_entry_off, val_entry_off.saturating_add(TYPE_LEN))?;
        let val_type: JsonType = type_bytes[0].try_into()?;
        let val_offset = self.try_decode_u32_le_at(val_entry_off + TYPE_LEN)? as usize;
        Ok(match val_type {
            JsonType::Literal => {
                let offset = val_entry_off + TYPE_LEN;
                let end = offset.checked_add(LITERAL_LEN).ok_or_else(|| {
                    Error::CorruptedData("JSON literal range overflow".into())
                })?;
                JsonRef::new(val_type, self.try_slice(offset, end)?)
            }
            JsonType::U64 | JsonType::I64 | JsonType::Double => {
                let end = val_offset.checked_add(NUMBER_LEN).ok_or_else(|| {
                    Error::CorruptedData("JSON number range overflow".into())
                })?;
                JsonRef::new(val_type, self.try_slice(val_offset, end)?)
            }
            JsonType::String => {
                let tail = self.value().get(val_offset..).ok_or_else(|| {
                    Error::CorruptedData(format!(
                        "JSON string offset {} past len {}",
                        val_offset,
                        self.value().len()
                    ))
                })?;
                let (str_len, len_len) = NumberCodec::try_decode_var_u64(tail)?;
                let total = (str_len as usize).checked_add(len_len).ok_or_else(|| {
                    Error::CorruptedData("JSON string length overflow".into())
                })?;
                let end = val_offset.checked_add(total).ok_or_else(|| {
                    Error::CorruptedData("JSON string range overflow".into())
                })?;
                JsonRef::new(val_type, self.try_slice(val_offset, end)?)
            }
            JsonType::Opaque => {
                let body_off = val_offset.checked_add(1).ok_or_else(|| {
                    Error::CorruptedData("JSON opaque offset overflow".into())
                })?;
                let tail = self.value().get(body_off..).ok_or_else(|| {
                    Error::CorruptedData(format!(
                        "JSON opaque offset {} past len {}",
                        body_off,
                        self.value().len()
                    ))
                })?;
                let (opaque_bytes_len, len_len) = NumberCodec::try_decode_var_u64(tail)?;
                let total = (opaque_bytes_len as usize)
                    .checked_add(len_len)
                    .and_then(|t| t.checked_add(1))
                    .ok_or_else(|| Error::CorruptedData("JSON opaque length overflow".into()))?;
                let end = val_offset.checked_add(total).ok_or_else(|| {
                    Error::CorruptedData("JSON opaque range overflow".into())
                })?;
                JsonRef::new(val_type, self.try_slice(val_offset, end)?)
            }
            JsonType::Date | JsonType::Datetime | JsonType::Timestamp => {
                let end = val_offset.checked_add(TIME_LEN).ok_or_else(|| {
                    Error::CorruptedData("JSON time range overflow".into())
                })?;
                JsonRef::new(val_type, self.try_slice(val_offset, end)?)
            }
            JsonType::Time => {
                let end = val_offset.checked_add(DURATION_LEN).ok_or_else(|| {
                    Error::CorruptedData("JSON duration range overflow".into())
                })?;
                JsonRef::new(val_type, self.try_slice(val_offset, end)?)
            }
            _ => {
                let size_off = val_offset.checked_add(ELEMENT_COUNT_LEN).ok_or_else(|| {
                    Error::CorruptedData("JSON nested size offset overflow".into())
                })?;
                let data_size = self.try_decode_u32_le_at(size_off)? as usize;
                let end = val_offset.checked_add(data_size).ok_or_else(|| {
                    Error::CorruptedData("JSON nested range overflow".into())
                })?;
                JsonRef::new(val_type, self.try_slice(val_offset, end)?)
            }
        })
    }

    /// Returns a raw pointer to the underlying values buffer.
    pub(super) fn as_ptr(&self) -> *const u8 {
        self.value.as_ptr()
    }

    /// Returns the literal value of JSON document
    pub(super) fn as_literal(&self) -> Result<u8> {
        match self.get_type() {
            JsonType::Literal => Ok(self.value()[0]),
            _ => Err(invalid_type!(
                "{} from {} to literal",
                ERR_CONVERT_FAILED,
                self.to_string_value()
            )),
        }
    }

    /// Returns the encoding binary length of self
    pub fn binary_len(&self) -> usize {
        TYPE_LEN + self.value.len()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::{
        codec::{
            data_type::Duration,
            mysql::{Json, Time, TimeType},
        },
        expr::EvalContext,
    };

    #[test]
    fn test_type() {
        let legal_cases = vec![
            (r#"{"key":"value"}"#, JsonType::Object),
            (r#"["d1","d2"]"#, JsonType::Array),
            (r#"-3"#, JsonType::I64),
            (r#"3"#, JsonType::I64),
            (r#"18446744073709551615"#, JsonType::U64),
            (r#"18446744073709551616"#, JsonType::Double),
            (r#"3.0"#, JsonType::Double),
            (r#"null"#, JsonType::Literal),
            (r#"true"#, JsonType::Literal),
            (r#"false"#, JsonType::Literal),
        ];

        for (json_str, tp) in legal_cases {
            let json: Json = json_str.parse().unwrap();
            assert_eq!(json.as_ref().get_type(), tp, "{:?}", json_str);
        }
    }

    #[test]
    fn test_array_get_elem() {
        let mut ctx = EvalContext::default();

        let time = Time::parse(
            &mut ctx,
            "1998-06-13 12:13:14",
            TimeType::DateTime,
            0,
            false,
        )
        .unwrap();
        let duration = Duration::parse(&mut ctx, "12:13:14", 0).unwrap();
        let array = vec![
            Json::from_u64(1).unwrap(),
            Json::from_str_val("abcdefg").unwrap(),
        ];
        let object = BTreeMap::from([
            ("key1".to_string(), Json::from_u64(1).unwrap()),
            ("key2".to_string(), Json::from_str_val("abcdefg").unwrap()),
        ]);

        let json_array = Json::from_array(vec![
            Json::from_u64(1).unwrap(),
            Json::from_time(time).unwrap(),
            Json::from_duration(duration).unwrap(),
            Json::from_array(array).unwrap(),
            Json::from_str_val("abcdefg").unwrap(),
            Json::from_bool(false).unwrap(),
            Json::from_object(object).unwrap(),
        ])
        .unwrap();
        let json_array_ref = json_array.as_ref();

        assert_eq!(json_array_ref.array_get_elem(0).unwrap().get_u64(), 1);
        assert_eq!(
            json_array_ref
                .array_get_elem(1)
                .unwrap()
                .get_time()
                .unwrap(),
            time
        );
        assert_eq!(
            json_array_ref
                .array_get_elem(2)
                .unwrap()
                .get_duration()
                .unwrap(),
            duration
        );
        assert_eq!(
            json_array_ref
                .array_get_elem(3)
                .unwrap()
                .array_get_elem(0)
                .unwrap()
                .get_u64(),
            1
        );
        assert_eq!(
            json_array_ref
                .array_get_elem(3)
                .unwrap()
                .array_get_elem(1)
                .unwrap()
                .get_str()
                .unwrap(),
            "abcdefg"
        );
        assert_eq!(
            json_array_ref.array_get_elem(4).unwrap().get_str().unwrap(),
            "abcdefg"
        );
        assert_eq!(
            json_array_ref
                .array_get_elem(5)
                .unwrap()
                .get_literal()
                .unwrap(),
            false
        );
        assert_eq!(
            json_array_ref
                .array_get_elem(6)
                .unwrap()
                .object_get_key(0)
                .unwrap(),
            b"key1"
        );
        assert_eq!(
            json_array_ref
                .array_get_elem(6)
                .unwrap()
                .object_get_key(1)
                .unwrap(),
            b"key2"
        );
        assert_eq!(
            json_array_ref
                .array_get_elem(6)
                .unwrap()
                .object_get_val(0)
                .unwrap()
                .get_u64(),
            1
        );
        assert_eq!(
            json_array_ref
                .array_get_elem(6)
                .unwrap()
                .object_get_val(1)
                .unwrap()
                .get_str()
                .unwrap(),
            "abcdefg"
        );
    }

    #[test]
    fn test_object_get_val() {
        let mut ctx = EvalContext::default();

        let time = Time::parse(
            &mut ctx,
            "1998-06-13 12:13:14",
            TimeType::DateTime,
            0,
            false,
        )
        .unwrap();
        let duration = Duration::parse(&mut ctx, "12:13:14", 0).unwrap();
        let array = vec![
            Json::from_u64(1).unwrap(),
            Json::from_str_val("abcdefg").unwrap(),
        ];
        let object = BTreeMap::from([
            ("key1".to_string(), Json::from_u64(1).unwrap()),
            ("key2".to_string(), Json::from_str_val("abcdefg").unwrap()),
        ]);

        let json_object = Json::from_object(BTreeMap::from([
            ("0".to_string(), Json::from_u64(1).unwrap()),
            ("1".to_string(), Json::from_time(time).unwrap()),
            ("2".to_string(), Json::from_duration(duration).unwrap()),
            ("3".to_string(), Json::from_array(array).unwrap()),
            ("4".to_string(), Json::from_str_val("abcdefg").unwrap()),
            ("5".to_string(), Json::from_bool(false).unwrap()),
            ("6".to_string(), Json::from_object(object).unwrap()),
        ]))
        .unwrap();
        let json_object_ref = json_object.as_ref();

        assert_eq!(json_object_ref.object_get_key(0).unwrap(), b"0");
        assert_eq!(json_object_ref.object_get_key(1).unwrap(), b"1");
        assert_eq!(json_object_ref.object_get_key(2).unwrap(), b"2");
        assert_eq!(json_object_ref.object_get_key(3).unwrap(), b"3");

        assert_eq!(json_object_ref.object_get_val(0).unwrap().get_u64(), 1);
        assert_eq!(
            json_object_ref
                .object_get_val(1)
                .unwrap()
                .get_time()
                .unwrap(),
            time
        );
        assert_eq!(
            json_object_ref
                .object_get_val(2)
                .unwrap()
                .get_duration()
                .unwrap(),
            duration
        );
        assert_eq!(
            json_object_ref
                .object_get_val(3)
                .unwrap()
                .array_get_elem(0)
                .unwrap()
                .get_u64(),
            1
        );
        assert_eq!(
            json_object_ref
                .object_get_val(3)
                .unwrap()
                .array_get_elem(1)
                .unwrap()
                .get_str()
                .unwrap(),
            "abcdefg"
        );
        assert_eq!(
            json_object_ref
                .object_get_val(4)
                .unwrap()
                .get_str()
                .unwrap(),
            "abcdefg"
        );
        assert_eq!(
            json_object_ref
                .object_get_val(5)
                .unwrap()
                .get_literal()
                .unwrap(),
            false
        );
        assert_eq!(
            json_object_ref
                .object_get_val(6)
                .unwrap()
                .object_get_key(0)
                .unwrap(),
            b"key1"
        );
        assert_eq!(
            json_object_ref
                .object_get_val(6)
                .unwrap()
                .object_get_key(1)
                .unwrap(),
            b"key2"
        );
        assert_eq!(
            json_object_ref
                .object_get_val(6)
                .unwrap()
                .object_get_val(0)
                .unwrap()
                .get_u64(),
            1
        );
        assert_eq!(
            json_object_ref
                .object_get_val(6)
                .unwrap()
                .object_get_val(1)
                .unwrap()
                .get_str()
                .unwrap(),
            "abcdefg"
        );
    }

    /// Poison / truncated object binary must return `CorruptedData`, not panic.
    ///
    /// Untrusted storage can present wild key offsets; slice indexing used to
    /// panic (coprocessor DoS). Bounds-checked accessors return an error.
    #[test]
    fn test_object_get_key_poison_offsets_err_not_panic() {
        // type Object + 8-byte header (elem_count=1, size=...) + key entry with
        // huge key_off so key slice is OOB.
        let mut value = vec![0u8; HEADER_LEN + KEY_ENTRY_LEN];
        let value_len = value.len() as u32;
        // element count = 1
        value[0..4].copy_from_slice(&1u32.to_le_bytes());
        // size = value.len()
        value[4..8].copy_from_slice(&value_len.to_le_bytes());
        // key_off = 0xffff_fff0, key_len = 4
        let key_entry = HEADER_LEN;
        value[key_entry..key_entry + 4].copy_from_slice(&0xffff_fff0u32.to_le_bytes());
        value[key_entry + 4..key_entry + 6].copy_from_slice(&4u16.to_le_bytes());

        let j = JsonRef::new(JsonType::Object, &value);
        let err = j.object_get_key(0).unwrap_err();
        match err {
            crate::codec::Error::CorruptedData(msg) => {
                assert!(
                    msg.contains("out of bounds") || msg.contains("overflow"),
                    "unexpected msg: {}",
                    msg
                );
            }
            other => panic!("expected CorruptedData, got {:?}", other),
        }

        // Truncated key-entry table: elem_count claims 2 keys but buffer too short.
        let mut short = vec![0u8; HEADER_LEN];
        short[0..4].copy_from_slice(&2u32.to_le_bytes());
        short[4..8].copy_from_slice(&8u32.to_le_bytes());
        let j2 = JsonRef::new(JsonType::Object, &short);
        assert!(j2.object_get_key(0).is_err());
        assert!(j2.object_search_key(b"x").is_err());
    }

    #[test]
    fn test_array_get_elem_truncated_err_not_panic() {
        // Array header claims 1 elem but no value entry bytes.
        let mut value = vec![0u8; HEADER_LEN];
        value[0..4].copy_from_slice(&1u32.to_le_bytes());
        value[4..8].copy_from_slice(&8u32.to_le_bytes());
        let j = JsonRef::new(JsonType::Array, &value);
        assert!(j.array_get_elem(0).is_err());
        assert!(j.val_entry_get(HEADER_LEN).is_err());
    }
}
