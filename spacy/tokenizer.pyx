from libc.stdint cimport uint64_t


cdef extern from *:
    r"""
    #include <stdint.h>
    #if defined(__SSE2__)
    #include <emmintrin.h>
    #endif
    static int rmcp_exact_unicode(PyObject *text) {
        return PyUnicode_CheckExact(text);
    }
    static unsigned rmcp_first_set_bit(uint64_t value) {
        #if defined(__GNUC__) || defined(__clang__)
        return (unsigned)__builtin_ctzll(value);
        #else
        unsigned result = 0;
        while (!(value & 1u)) { ++result; value >>= 1; }
        return result;
        #endif
    }
    #if defined(__SSE2__)
    static __m128i rmcp_space8(__m128i c) {
        __m128i a = _mm_and_si128(_mm_cmpgt_epi8(c, _mm_set1_epi8(8)),
                                 _mm_cmpgt_epi8(_mm_set1_epi8(14), c));
        __m128i b = _mm_and_si128(_mm_cmpgt_epi8(c, _mm_set1_epi8(27)),
                                 _mm_cmpgt_epi8(_mm_set1_epi8(33), c));
        __m128i d = _mm_or_si128(_mm_cmpeq_epi8(c, _mm_set1_epi8((char)0x85)),
                                _mm_cmpeq_epi8(c, _mm_set1_epi8((char)0xa0)));
        return _mm_or_si128(_mm_or_si128(a, b), d);
    }
    static __m128i rmcp_space16(__m128i c) {
        __m128i a = _mm_and_si128(_mm_cmpgt_epi16(c, _mm_set1_epi16(8)),
                                 _mm_cmpgt_epi16(_mm_set1_epi16(14), c));
        __m128i b = _mm_and_si128(_mm_cmpgt_epi16(c, _mm_set1_epi16(27)),
                                 _mm_cmpgt_epi16(_mm_set1_epi16(33), c));
        __m128i d = _mm_or_si128(_mm_cmpeq_epi16(c, _mm_set1_epi16(0x85)),
                                _mm_cmpeq_epi16(c, _mm_set1_epi16(0xa0)));
        __m128i e = _mm_or_si128(_mm_cmpeq_epi16(c, _mm_set1_epi16(0x1680)),
                                _mm_cmpeq_epi16(c, _mm_set1_epi16(0x202f)));
        __m128i f = _mm_or_si128(_mm_cmpeq_epi16(c, _mm_set1_epi16(0x205f)),
                                _mm_cmpeq_epi16(c, _mm_set1_epi16(0x3000)));
        __m128i g = _mm_and_si128(_mm_cmpgt_epi16(c, _mm_set1_epi16(0x1fff)),
                                 _mm_cmpgt_epi16(_mm_set1_epi16(0x200b), c));
        __m128i h = _mm_cmpeq_epi16(_mm_and_si128(c, _mm_set1_epi16((short)~1)),
                                   _mm_set1_epi16(0x2028));
        return _mm_or_si128(_mm_or_si128(_mm_or_si128(a, b), _mm_or_si128(d, e)),
                            _mm_or_si128(_mm_or_si128(f, g), h));
    }
    static __m128i rmcp_space32(__m128i c) {
        __m128i a = _mm_and_si128(_mm_cmpgt_epi32(c, _mm_set1_epi32(8)),
                                 _mm_cmpgt_epi32(_mm_set1_epi32(14), c));
        __m128i b = _mm_and_si128(_mm_cmpgt_epi32(c, _mm_set1_epi32(27)),
                                 _mm_cmpgt_epi32(_mm_set1_epi32(33), c));
        __m128i d = _mm_or_si128(_mm_cmpeq_epi32(c, _mm_set1_epi32(0x85)),
                                _mm_cmpeq_epi32(c, _mm_set1_epi32(0xa0)));
        __m128i e = _mm_or_si128(_mm_cmpeq_epi32(c, _mm_set1_epi32(0x1680)),
                                _mm_cmpeq_epi32(c, _mm_set1_epi32(0x202f)));
        __m128i f = _mm_or_si128(_mm_cmpeq_epi32(c, _mm_set1_epi32(0x205f)),
                                _mm_cmpeq_epi32(c, _mm_set1_epi32(0x3000)));
        __m128i g = _mm_and_si128(_mm_cmpgt_epi32(c, _mm_set1_epi32(0x1fff)),
                                 _mm_cmpgt_epi32(_mm_set1_epi32(0x200b), c));
        __m128i h = _mm_cmpeq_epi32(_mm_and_si128(c, _mm_set1_epi32(~1)),
                                   _mm_set1_epi32(0x2028));
        return _mm_or_si128(_mm_or_si128(_mm_or_si128(a, b), _mm_or_si128(d, e)),
                            _mm_or_si128(_mm_or_si128(f, g), h));
    }
    #endif
    static uint64_t rmcp_space_transition_block(PyObject *text, Py_ssize_t start,
                                                unsigned width, int previous_space) {
        const int kind = PyUnicode_KIND(text);
        const void *data = PyUnicode_DATA(text);
        uint64_t spaces = 0;
        unsigned pos = 0;
        #if defined(__SSE2__)
        const unsigned lanes = 16 / kind;
        while (width - pos >= lanes) {
            const unsigned char *bytes = (const unsigned char *)data + (start + pos) * kind;
            const __m128i c = _mm_loadu_si128((const __m128i *)bytes);
            const __m128i matches = kind == 1 ? rmcp_space8(c) : kind == 2 ? rmcp_space16(c) : rmcp_space32(c);
            unsigned bits;
            if (kind == 4) {
                bits = (unsigned)_mm_movemask_ps(_mm_castsi128_ps(matches));
            } else {
                bits = (unsigned)_mm_movemask_epi8(matches);
                if (kind == 2) {
                    bits &= 0x5555u;
                    bits = (bits | (bits >> 1)) & 0x3333u;
                    bits = (bits | (bits >> 2)) & 0x0f0fu;
                    bits = (bits | (bits >> 4)) & 0x00ffu;
                }
            }
            spaces |= (uint64_t)bits << pos;
            pos += lanes;
        }
        #endif
        for (; pos < width; ++pos) {
            spaces |= (uint64_t)!!Py_UNICODE_ISSPACE(PyUnicode_READ(kind, data, start + pos)) << pos;
        }
        const uint64_t valid = width == 64 ? UINT64_MAX : (((uint64_t)1 << width) - 1);
        return (spaces ^ ((spaces << 1) | (uint64_t)previous_space)) & valid;
    }
    """
    bint rmcp_exact_unicode(object text)
    unsigned rmcp_first_set_bit(uint64_t value)
    uint64_t rmcp_space_transition_block(object text, Py_ssize_t start, unsigned width, int previous_space)

# cython: embedsignature=True, binding=True
cimport cython
from cymem.cymem cimport Pool
from cython.operator cimport dereference as deref
from cython.operator cimport preincrement as preinc
from libc.string cimport memcpy, memset
from libcpp.set cimport set as stdset
from preshed.maps cimport PreshMap

import re
from .lexeme cimport EMPTY_LEXEME
from .strings cimport hash_string
from .tokens.doc cimport Doc

from . import util
from .attrs import intify_attrs
from .errors import Errors
from .scorer import Scorer
from .symbols import NORM, ORTH
from .tokens import Span
from .training import validate_examples
from .util import get_words_and_spaces


from re import Pattern as _RegexPattern

cdef extern from *:
    r"""
    static PyCFunction rmcp_ascii_regex_match_function = NULL;
    static void rmcp_set_ascii_regex_match_function(PyObject *callback) {
        rmcp_ascii_regex_match_function = PyCFunction_GET_FUNCTION(callback);
    }
    static int rmcp_ascii_word_text(PyObject *text) {
        if (!PyUnicode_CheckExact(text) || !PyUnicode_IS_ASCII(text)) return 0;
        Py_ssize_t length = PyUnicode_GET_LENGTH(text);
        if (!length) return 0;
        const unsigned char *data = PyUnicode_1BYTE_DATA(text);
        for (Py_ssize_t i = 0; i < length; ++i) {
            unsigned int lower = data[i] | 32u;
            if (lower < 'a' || lower > 'z') return 0;
        }
        return 1;
    }
    static int rmcp_exact_ascii_pattern(PyObject *callback, PyObject *pattern_type) {
        if (!PyCFunction_Check(callback) ||
            PyCFunction_GET_FUNCTION(callback) != rmcp_ascii_regex_match_function)
            return 0;
        PyObject *owner = PyCFunction_GET_SELF(callback);
        if (owner == NULL || !Py_IS_TYPE(owner, (PyTypeObject *)pattern_type))
            return 0;
        PyObject *pattern = PyObject_GetAttrString(owner, "pattern");
        if (pattern == NULL) return -1;
        int equal = PyUnicode_CheckExact(pattern) &&
            PyUnicode_CompareWithASCIIString(pattern, "[A-Za-z]+\\Z") == 0;
        Py_DECREF(pattern);
        if (!equal) return 0;
        PyObject *flags = PyObject_GetAttrString(owner, "flags");
        if (flags == NULL) return -1;
        long value = PyLong_AsLong(flags);
        Py_DECREF(flags);
        if (value == -1 && PyErr_Occurred()) return -1;
        return value == 32;
    }
    """
    void rmcp_set_ascii_regex_match_function(object callback)
    bint rmcp_ascii_word_text(object text)
    int rmcp_exact_ascii_pattern(object callback, object pattern_type) except -1

rmcp_set_ascii_regex_match_function(re.compile('').match)


cdef str _rmcp_default_english_suffix_pattern = '…$|……$|,$|:$|;$|\\!$|\\?$|¿$|؟$|¡$|\\($|\\)$|\\[$|\\]$|\\{$|\\}$|<$|>$|_$|#$|\\*$|&$|。$|？$|！$|，$|、$|；$|：$|～$|·$|।$|،$|۔$|؛$|٪$|\\.\\.+$|…$|\\\'$|"$|”$|“$|`$|‘$|´$|’$|‚$|,$|„$|»$|«$|「$|」$|『$|』$|（$|）$|〔$|〕$|【$|】$|《$|》$|〈$|〉$|〈$|〉$|⟦$|⟧$|[\\u00A6\\u00A9\\u00AE\\u00B0\\u0482\\u058D\\u058E\\u060E\\u060F\\u06DE\\u06E9\\u06FD\\u06FE\\u07F6\\u09FA\\u0B70\\u0BF3-\\u0BF8\\u0BFA\\u0C7F\\u0D4F\\u0D79\\u0F01-\\u0F03\\u0F13\\u0F15-\\u0F17\\u0F1A-\\u0F1F\\u0F34\\u0F36\\u0F38\\u0FBE-\\u0FC5\\u0FC7-\\u0FCC\\u0FCE\\u0FCF\\u0FD5-\\u0FD8\\u109E\\u109F\\u1390-\\u1399\\u1940\\u19DE-\\u19FF\\u1B61-\\u1B6A\\u1B74-\\u1B7C\\u2100\\u2101\\u2103-\\u2106\\u2108\\u2109\\u2114\\u2116\\u2117\\u211E-\\u2123\\u2125\\u2127\\u2129\\u212E\\u213A\\u213B\\u214A\\u214C\\u214D\\u214F\\u218A\\u218B\\u2195-\\u2199\\u219C-\\u219F\\u21A1\\u21A2\\u21A4\\u21A5\\u21A7-\\u21AD\\u21AF-\\u21CD\\u21D0\\u21D1\\u21D3\\u21D5-\\u21F3\\u2300-\\u2307\\u230C-\\u231F\\u2322-\\u2328\\u232B-\\u237B\\u237D-\\u239A\\u23B4-\\u23DB\\u23E2-\\u2426\\u2440-\\u244A\\u249C-\\u24E9\\u2500-\\u25B6\\u25B8-\\u25C0\\u25C2-\\u25F7\\u2600-\\u266E\\u2670-\\u2767\\u2794-\\u27BF\\u2800-\\u28FF\\u2B00-\\u2B2F\\u2B45\\u2B46\\u2B4D-\\u2B73\\u2B76-\\u2B95\\u2B98-\\u2BC8\\u2BCA-\\u2BFE\\u2CE5-\\u2CEA\\u2E80-\\u2E99\\u2E9B-\\u2EF3\\u2F00-\\u2FD5\\u2FF0-\\u2FFB\\u3004\\u3012\\u3013\\u3020\\u3036\\u3037\\u303E\\u303F\\u3190\\u3191\\u3196-\\u319F\\u31C0-\\u31E3\\u3200-\\u321E\\u322A-\\u3247\\u3250\\u3260-\\u327F\\u328A-\\u32B0\\u32C0-\\u32FE\\u3300-\\u33FF\\u4DC0-\\u4DFF\\uA490-\\uA4C6\\uA828-\\uA82B\\uA836\\uA837\\uA839\\uAA77-\\uAA79\\uFDFD\\uFFE4\\uFFE8\\uFFED\\uFFEE\\uFFFC\\uFFFD\\U00010137-\\U0001013F\\U00010179-\\U00010189\\U0001018C-\\U0001018E\\U00010190-\\U0001019B\\U000101A0\\U000101D0-\\U000101FC\\U00010877\\U00010878\\U00010AC8\\U0001173F\\U00016B3C-\\U00016B3F\\U00016B45\\U0001BC9C\\U0001D000-\\U0001D0F5\\U0001D100-\\U0001D126\\U0001D129-\\U0001D164\\U0001D16A-\\U0001D16C\\U0001D183\\U0001D184\\U0001D18C-\\U0001D1A9\\U0001D1AE-\\U0001D1E8\\U0001D200-\\U0001D241\\U0001D245\\U0001D300-\\U0001D356\\U0001D800-\\U0001D9FF\\U0001DA37-\\U0001DA3A\\U0001DA6D-\\U0001DA74\\U0001DA76-\\U0001DA83\\U0001DA85\\U0001DA86\\U0001ECAC\\U0001F000-\\U0001F02B\\U0001F030-\\U0001F093\\U0001F0A0-\\U0001F0AE\\U0001F0B1-\\U0001F0BF\\U0001F0C1-\\U0001F0CF\\U0001F0D1-\\U0001F0F5\\U0001F110-\\U0001F16B\\U0001F170-\\U0001F1AC\\U0001F1E6-\\U0001F202\\U0001F210-\\U0001F23B\\U0001F240-\\U0001F248\\U0001F250\\U0001F251\\U0001F260-\\U0001F265\\U0001F300-\\U0001F3FA\\U0001F400-\\U0001F6D4\\U0001F6E0-\\U0001F6EC\\U0001F6F0-\\U0001F6F9\\U0001F700-\\U0001F773\\U0001F780-\\U0001F7D8\\U0001F800-\\U0001F80B\\U0001F810-\\U0001F847\\U0001F850-\\U0001F859\\U0001F860-\\U0001F887\\U0001F890-\\U0001F8AD\\U0001F900-\\U0001F90B\\U0001F910-\\U0001F93E\\U0001F940-\\U0001F970\\U0001F973-\\U0001F976\\U0001F97A\\U0001F97C-\\U0001F9A2\\U0001F9B0-\\U0001F9B9\\U0001F9C0-\\U0001F9C2\\U0001F9D0-\\U0001F9FF\\U0001FA60-\\U0001FA6D]$|\'s$|\'S$|’s$|’S$|—$|–$|(?<=[0-9])\\+$|(?<=°[FfCcKk])\\.$|(?<=[0-9])(?:\\$|£|€|¥|฿|US\\$|C\\$|A\\$|₽|﷼|₴|₠|₡|₢|₣|₤|₥|₦|₧|₨|₩|₪|₫|€|₭|₮|₯|₰|₱|₲|₳|₴|₵|₶|₷|₸|₹|₺|₻|₼|₽|₾|₿)$|(?<=[0-9])(?:km|km²|km³|m|m²|m³|dm|dm²|dm³|cm|cm²|cm³|mm|mm²|mm³|ha|µm|nm|yd|in|ft|kg|g|mg|µg|t|lb|oz|m/s|km/h|kmh|mph|hPa|Pa|mbar|mb|MB|kb|KB|gb|GB|tb|TB|T|G|M|K|%|км|км²|км³|м|м²|м³|дм|дм²|дм³|см|см²|см³|мм|мм²|мм³|нм|кг|г|мг|м/с|км/ч|кПа|Па|мбар|Кб|КБ|кб|Мб|МБ|мб|Гб|ГБ|гб|Тб|ТБ|тбكم|كم²|كم³|م|م²|م³|سم|سم²|سم³|مم|مم²|مم³|كم|غرام|جرام|جم|كغ|ملغ|كوب|اكواب)$|(?<=[0-9a-z\\uFF41-\\uFF5A\\u00DF-\\u00F6\\u00F8-\\u00FF\\u0101\\u0103\\u0105\\u0107\\u0109\\u010B\\u010D\\u010F\\u0111\\u0113\\u0115\\u0117\\u0119\\u011B\\u011D\\u011F\\u0121\\u0123\\u0125\\u0127\\u0129\\u012B\\u012D\\u012F\\u0131\\u0133\\u0135\\u0137\\u0138\\u013A\\u013C\\u013E\\u0140\\u0142\\u0144\\u0146\\u0148\\u0149\\u014B\\u014D\\u014F\\u0151\\u0153\\u0155\\u0157\\u0159\\u015B\\u015D\\u015F\\u0161\\u0163\\u0165\\u0167\\u0169\\u016B\\u016D\\u016F\\u0171\\u0173\\u0175\\u0177\\u017A\\u017C\\u017E\\u017F\\u0180\\u0183\\u0185\\u0188\\u018C\\u018D\\u0192\\u0195\\u0199-\\u019B\\u019E\\u01A1\\u01A3\\u01A5\\u01A8\\u01AA\\u01AB\\u01AD\\u01B0\\u01B4\\u01B6\\u01B9\\u01BA\\u01BD-\\u01BF\\u01C6\\u01C9\\u01CC\\u01CE\\u01D0\\u01D2\\u01D4\\u01D6\\u01D8\\u01DA\\u01DC\\u01DD\\u01DF\\u01E1\\u01E3\\u01E5\\u01E7\\u01E9\\u01EB\\u01ED\\u01EF\\u01F0\\u01F3\\u01F5\\u01F9\\u01FB\\u01FD\\u01FF\\u0201\\u0203\\u0205\\u0207\\u0209\\u020B\\u020D\\u020F\\u0211\\u0213\\u0215\\u0217\\u0219\\u021B\\u021D\\u021F\\u0221\\u0223\\u0225\\u0227\\u0229\\u022B\\u022D\\u022F\\u0231\\u0233-\\u0239\\u023C\\u023F\\u0240\\u0242\\u0247\\u0249\\u024B\\u024D\\u024F\\u2C61\\u2C65\\u2C66\\u2C68\\u2C6A\\u2C6C\\u2C71\\u2C73\\u2C74\\u2C76-\\u2C7B\\uA723\\uA725\\uA727\\uA729\\uA72B\\uA72D\\uA72F-\\uA731\\uA733\\uA735\\uA737\\uA739\\uA73B\\uA73D\\uA73F\\uA741\\uA743\\uA745\\uA747\\uA749\\uA74B\\uA74D\\uA74F\\uA751\\uA753\\uA755\\uA757\\uA759\\uA75B\\uA75D\\uA75F\\uA761\\uA763\\uA765\\uA767\\uA769\\uA76B\\uA76D\\uA76F\\uA771-\\uA778\\uA77A\\uA77C\\uA77F\\uA781\\uA783\\uA785\\uA787\\uA78C\\uA78E\\uA791\\uA793-\\uA795\\uA797\\uA799\\uA79B\\uA79D\\uA79F\\uA7A1\\uA7A3\\uA7A5\\uA7A7\\uA7A9\\uA7AF\\uA7B5\\uA7B7\\uA7B9\\uA7FA\\uAB30-\\uAB5A\\uAB60-\\uAB64\\u0250-\\u02AF\\u1D00-\\u1D25\\u1D6B-\\u1D77\\u1D79-\\u1D9A\\u1E01\\u1E03\\u1E05\\u1E07\\u1E09\\u1E0B\\u1E0D\\u1E0F\\u1E11\\u1E13\\u1E15\\u1E17\\u1E19\\u1E1B\\u1E1D\\u1E1F\\u1E21\\u1E23\\u1E25\\u1E27\\u1E29\\u1E2B\\u1E2D\\u1E2F\\u1E31\\u1E33\\u1E35\\u1E37\\u1E39\\u1E3B\\u1E3D\\u1E3F\\u1E41\\u1E43\\u1E45\\u1E47\\u1E49\\u1E4B\\u1E4D\\u1E4F\\u1E51\\u1E53\\u1E55\\u1E57\\u1E59\\u1E5B\\u1E5D\\u1E5F\\u1E61\\u1E63\\u1E65\\u1E67\\u1E69\\u1E6B\\u1E6D\\u1E6F\\u1E71\\u1E73\\u1E75\\u1E77\\u1E79\\u1E7B\\u1E7D\\u1E7F\\u1E81\\u1E83\\u1E85\\u1E87\\u1E89\\u1E8B\\u1E8D\\u1E8F\\u1E91\\u1E93\\u1E95-\\u1E9D\\u1E9F\\u1EA1\\u1EA3\\u1EA5\\u1EA7\\u1EA9\\u1EAB\\u1EAD\\u1EAF\\u1EB1\\u1EB3\\u1EB5\\u1EB7\\u1EB9\\u1EBB\\u1EBD\\u1EBF\\u1EC1\\u1EC3\\u1EC5\\u1EC7\\u1EC9\\u1ECB\\u1ECD\\u1ECF\\u1ED1\\u1ED3\\u1ED5\\u1ED7\\u1ED9\\u1EDB\\u1EDD\\u1EDF\\u1EE1\\u1EE3\\u1EE5\\u1EE7\\u1EE9\\u1EEB\\u1EED\\u1EEF\\u1EF1\\u1EF3\\u1EF5\\u1EF7\\u1EF9\\u1EFB\\u1EFD\\u1EFFёа-яәөүҗңһα-ωάέίόώήύа-щюяіїєґѓѕјљњќѐѝ\\u1200-\\u137F\\u0980-\\u09FF\\u0591-\\u05F4\\uFB1D-\\uFB4F\\u0620-\\u064A\\u066E-\\u06D5\\u06E5-\\u06FF\\u0750-\\u077F\\u08A0-\\u08BD\\uFB50-\\uFBB1\\uFBD3-\\uFD3D\\uFD50-\\uFDC7\\uFDF0-\\uFDFB\\uFE70-\\uFEFC\\U0001EE00-\\U0001EEBB\\u0D80-\\u0DFF\\u0900-\\u097F\\u0C80-\\u0CFF\\u0B80-\\u0BFF\\u0C00-\\u0C7F\\uAC00-\\uD7AF\\u1100-\\u11FF\\u3040-\\u309F\\u30A0-\\u30FFー\\u4E00-\\u62FF\\u6300-\\u77FF\\u7800-\\u8CFF\\u8D00-\\u9FFF\\u3400-\\u4DBF\\U00020000-\\U000215FF\\U00021600-\\U000230FF\\U00023100-\\U000245FF\\U00024600-\\U000260FF\\U00026100-\\U000275FF\\U00027600-\\U000290FF\\U00029100-\\U0002A6DF\\U0002A700-\\U0002B73F\\U0002B740-\\U0002B81F\\U0002B820-\\U0002CEAF\\U0002CEB0-\\U0002EBEF\\u2E80-\\u2EFF\\u2F00-\\u2FDF\\u2FF0-\\u2FFF\\u3000-\\u303F\\u31C0-\\u31EF\\u3200-\\u32FF\\u3300-\\u33FF\\uF900-\\uFAFF\\uFE30-\\uFE4F\\U0001F200-\\U0001F2FF\\U0002F800-\\U0002FA1F%²\\-\\+…|……|,|:|;|\\!|\\?|¿|؟|¡|\\(|\\)|\\[|\\]|\\{|\\}|<|>|_|#|\\*|&|。|？|！|，|、|；|：|～|·|।|،|۔|؛|٪(?:\\\'"”“`‘´’‚,„»«「」『』（）〔〕【】《》〈〉〈〉⟦⟧)])\\.$|(?<=[A-Z\\uFF21-\\uFF3A\\u00C0-\\u00D6\\u00D8-\\u00DE\\u0100\\u0102\\u0104\\u0106\\u0108\\u010A\\u010C\\u010E\\u0110\\u0112\\u0114\\u0116\\u0118\\u011A\\u011C\\u011E\\u0120\\u0122\\u0124\\u0126\\u0128\\u012A\\u012C\\u012E\\u0130\\u0132\\u0134\\u0136\\u0139\\u013B\\u013D\\u013F\\u0141\\u0143\\u0145\\u0147\\u014A\\u014C\\u014E\\u0150\\u0152\\u0154\\u0156\\u0158\\u015A\\u015C\\u015E\\u0160\\u0162\\u0164\\u0166\\u0168\\u016A\\u016C\\u016E\\u0170\\u0172\\u0174\\u0176\\u0178\\u0179\\u017B\\u017D\\u0181\\u0182\\u0184\\u0186\\u0187\\u0189-\\u018B\\u018E-\\u0191\\u0193\\u0194\\u0196-\\u0198\\u019C\\u019D\\u019F\\u01A0\\u01A2\\u01A4\\u01A6\\u01A7\\u01A9\\u01AC\\u01AE\\u01AF\\u01B1-\\u01B3\\u01B5\\u01B7\\u01B8\\u01BC\\u01C4\\u01C7\\u01CA\\u01CD\\u01CF\\u01D1\\u01D3\\u01D5\\u01D7\\u01D9\\u01DB\\u01DE\\u01E0\\u01E2\\u01E4\\u01E6\\u01E8\\u01EA\\u01EC\\u01EE\\u01F1\\u01F4\\u01F6-\\u01F8\\u01FA\\u01FC\\u01FE\\u0200\\u0202\\u0204\\u0206\\u0208\\u020A\\u020C\\u020E\\u0210\\u0212\\u0214\\u0216\\u0218\\u021A\\u021C\\u021E\\u0220\\u0222\\u0224\\u0226\\u0228\\u022A\\u022C\\u022E\\u0230\\u0232\\u023A\\u023B\\u023D\\u023E\\u0241\\u0243-\\u0246\\u0248\\u024A\\u024C\\u024E\\u2C60\\u2C62-\\u2C64\\u2C67\\u2C69\\u2C6B\\u2C6D-\\u2C70\\u2C72\\u2C75\\u2C7E\\u2C7F\\uA722\\uA724\\uA726\\uA728\\uA72A\\uA72C\\uA72E\\uA732\\uA734\\uA736\\uA738\\uA73A\\uA73C\\uA73E\\uA740\\uA742\\uA744\\uA746\\uA748\\uA74A\\uA74C\\uA74E\\uA750\\uA752\\uA754\\uA756\\uA758\\uA75A\\uA75C\\uA75E\\uA760\\uA762\\uA764\\uA766\\uA768\\uA76A\\uA76C\\uA76E\\uA779\\uA77B\\uA77D\\uA77E\\uA780\\uA782\\uA784\\uA786\\uA78B\\uA78D\\uA790\\uA792\\uA796\\uA798\\uA79A\\uA79C\\uA79E\\uA7A0\\uA7A2\\uA7A4\\uA7A6\\uA7A8\\uA7AA-\\uA7AE\\uA7B0-\\uA7B4\\uA7B6\\uA7B8\\u1E00\\u1E02\\u1E04\\u1E06\\u1E08\\u1E0A\\u1E0C\\u1E0E\\u1E10\\u1E12\\u1E14\\u1E16\\u1E18\\u1E1A\\u1E1C\\u1E1E\\u1E20\\u1E22\\u1E24\\u1E26\\u1E28\\u1E2A\\u1E2C\\u1E2E\\u1E30\\u1E32\\u1E34\\u1E36\\u1E38\\u1E3A\\u1E3C\\u1E3E\\u1E40\\u1E42\\u1E44\\u1E46\\u1E48\\u1E4A\\u1E4C\\u1E4E\\u1E50\\u1E52\\u1E54\\u1E56\\u1E58\\u1E5A\\u1E5C\\u1E5E\\u1E60\\u1E62\\u1E64\\u1E66\\u1E68\\u1E6A\\u1E6C\\u1E6E\\u1E70\\u1E72\\u1E74\\u1E76\\u1E78\\u1E7A\\u1E7C\\u1E7E\\u1E80\\u1E82\\u1E84\\u1E86\\u1E88\\u1E8A\\u1E8C\\u1E8E\\u1E90\\u1E92\\u1E94\\u1E9E\\u1EA0\\u1EA2\\u1EA4\\u1EA6\\u1EA8\\u1EAA\\u1EAC\\u1EAE\\u1EB0\\u1EB2\\u1EB4\\u1EB6\\u1EB8\\u1EBA\\u1EBC\\u1EBE\\u1EC0\\u1EC2\\u1EC4\\u1EC6\\u1EC8\\u1ECA\\u1ECC\\u1ECE\\u1ED0\\u1ED2\\u1ED4\\u1ED6\\u1ED8\\u1EDA\\u1EDC\\u1EDE\\u1EE0\\u1EE2\\u1EE4\\u1EE6\\u1EE8\\u1EEA\\u1EEC\\u1EEE\\u1EF0\\u1EF2\\u1EF4\\u1EF6\\u1EF8\\u1EFA\\u1EFC\\u1EFEЁА-ЯӘӨҮҖҢҺΑ-ΩΆΈΊΌΏΉΎА-ЩЮЯІЇЄҐЃЅЈЉЊЌЀЍ\\u1200-\\u137F\\u0980-\\u09FF\\u0591-\\u05F4\\uFB1D-\\uFB4F\\u0620-\\u064A\\u066E-\\u06D5\\u06E5-\\u06FF\\u0750-\\u077F\\u08A0-\\u08BD\\uFB50-\\uFBB1\\uFBD3-\\uFD3D\\uFD50-\\uFDC7\\uFDF0-\\uFDFB\\uFE70-\\uFEFC\\U0001EE00-\\U0001EEBB\\u0D80-\\u0DFF\\u0900-\\u097F\\u0C80-\\u0CFF\\u0B80-\\u0BFF\\u0C00-\\u0C7F\\uAC00-\\uD7AF\\u1100-\\u11FF\\u3040-\\u309F\\u30A0-\\u30FFー\\u4E00-\\u62FF\\u6300-\\u77FF\\u7800-\\u8CFF\\u8D00-\\u9FFF\\u3400-\\u4DBF\\U00020000-\\U000215FF\\U00021600-\\U000230FF\\U00023100-\\U000245FF\\U00024600-\\U000260FF\\U00026100-\\U000275FF\\U00027600-\\U000290FF\\U00029100-\\U0002A6DF\\U0002A700-\\U0002B73F\\U0002B740-\\U0002B81F\\U0002B820-\\U0002CEAF\\U0002CEB0-\\U0002EBEF\\u2E80-\\u2EFF\\u2F00-\\u2FDF\\u2FF0-\\u2FFF\\u3000-\\u303F\\u31C0-\\u31EF\\u3200-\\u32FF\\u3300-\\u33FF\\uF900-\\uFAFF\\uFE30-\\uFE4F\\U0001F200-\\U0001F2FF\\U0002F800-\\U0002FA1F][A-Z\\uFF21-\\uFF3A\\u00C0-\\u00D6\\u00D8-\\u00DE\\u0100\\u0102\\u0104\\u0106\\u0108\\u010A\\u010C\\u010E\\u0110\\u0112\\u0114\\u0116\\u0118\\u011A\\u011C\\u011E\\u0120\\u0122\\u0124\\u0126\\u0128\\u012A\\u012C\\u012E\\u0130\\u0132\\u0134\\u0136\\u0139\\u013B\\u013D\\u013F\\u0141\\u0143\\u0145\\u0147\\u014A\\u014C\\u014E\\u0150\\u0152\\u0154\\u0156\\u0158\\u015A\\u015C\\u015E\\u0160\\u0162\\u0164\\u0166\\u0168\\u016A\\u016C\\u016E\\u0170\\u0172\\u0174\\u0176\\u0178\\u0179\\u017B\\u017D\\u0181\\u0182\\u0184\\u0186\\u0187\\u0189-\\u018B\\u018E-\\u0191\\u0193\\u0194\\u0196-\\u0198\\u019C\\u019D\\u019F\\u01A0\\u01A2\\u01A4\\u01A6\\u01A7\\u01A9\\u01AC\\u01AE\\u01AF\\u01B1-\\u01B3\\u01B5\\u01B7\\u01B8\\u01BC\\u01C4\\u01C7\\u01CA\\u01CD\\u01CF\\u01D1\\u01D3\\u01D5\\u01D7\\u01D9\\u01DB\\u01DE\\u01E0\\u01E2\\u01E4\\u01E6\\u01E8\\u01EA\\u01EC\\u01EE\\u01F1\\u01F4\\u01F6-\\u01F8\\u01FA\\u01FC\\u01FE\\u0200\\u0202\\u0204\\u0206\\u0208\\u020A\\u020C\\u020E\\u0210\\u0212\\u0214\\u0216\\u0218\\u021A\\u021C\\u021E\\u0220\\u0222\\u0224\\u0226\\u0228\\u022A\\u022C\\u022E\\u0230\\u0232\\u023A\\u023B\\u023D\\u023E\\u0241\\u0243-\\u0246\\u0248\\u024A\\u024C\\u024E\\u2C60\\u2C62-\\u2C64\\u2C67\\u2C69\\u2C6B\\u2C6D-\\u2C70\\u2C72\\u2C75\\u2C7E\\u2C7F\\uA722\\uA724\\uA726\\uA728\\uA72A\\uA72C\\uA72E\\uA732\\uA734\\uA736\\uA738\\uA73A\\uA73C\\uA73E\\uA740\\uA742\\uA744\\uA746\\uA748\\uA74A\\uA74C\\uA74E\\uA750\\uA752\\uA754\\uA756\\uA758\\uA75A\\uA75C\\uA75E\\uA760\\uA762\\uA764\\uA766\\uA768\\uA76A\\uA76C\\uA76E\\uA779\\uA77B\\uA77D\\uA77E\\uA780\\uA782\\uA784\\uA786\\uA78B\\uA78D\\uA790\\uA792\\uA796\\uA798\\uA79A\\uA79C\\uA79E\\uA7A0\\uA7A2\\uA7A4\\uA7A6\\uA7A8\\uA7AA-\\uA7AE\\uA7B0-\\uA7B4\\uA7B6\\uA7B8\\u1E00\\u1E02\\u1E04\\u1E06\\u1E08\\u1E0A\\u1E0C\\u1E0E\\u1E10\\u1E12\\u1E14\\u1E16\\u1E18\\u1E1A\\u1E1C\\u1E1E\\u1E20\\u1E22\\u1E24\\u1E26\\u1E28\\u1E2A\\u1E2C\\u1E2E\\u1E30\\u1E32\\u1E34\\u1E36\\u1E38\\u1E3A\\u1E3C\\u1E3E\\u1E40\\u1E42\\u1E44\\u1E46\\u1E48\\u1E4A\\u1E4C\\u1E4E\\u1E50\\u1E52\\u1E54\\u1E56\\u1E58\\u1E5A\\u1E5C\\u1E5E\\u1E60\\u1E62\\u1E64\\u1E66\\u1E68\\u1E6A\\u1E6C\\u1E6E\\u1E70\\u1E72\\u1E74\\u1E76\\u1E78\\u1E7A\\u1E7C\\u1E7E\\u1E80\\u1E82\\u1E84\\u1E86\\u1E88\\u1E8A\\u1E8C\\u1E8E\\u1E90\\u1E92\\u1E94\\u1E9E\\u1EA0\\u1EA2\\u1EA4\\u1EA6\\u1EA8\\u1EAA\\u1EAC\\u1EAE\\u1EB0\\u1EB2\\u1EB4\\u1EB6\\u1EB8\\u1EBA\\u1EBC\\u1EBE\\u1EC0\\u1EC2\\u1EC4\\u1EC6\\u1EC8\\u1ECA\\u1ECC\\u1ECE\\u1ED0\\u1ED2\\u1ED4\\u1ED6\\u1ED8\\u1EDA\\u1EDC\\u1EDE\\u1EE0\\u1EE2\\u1EE4\\u1EE6\\u1EE8\\u1EEA\\u1EEC\\u1EEE\\u1EF0\\u1EF2\\u1EF4\\u1EF6\\u1EF8\\u1EFA\\u1EFC\\u1EFEЁА-ЯӘӨҮҖҢҺΑ-ΩΆΈΊΌΏΉΎА-ЩЮЯІЇЄҐЃЅЈЉЊЌЀЍ\\u1200-\\u137F\\u0980-\\u09FF\\u0591-\\u05F4\\uFB1D-\\uFB4F\\u0620-\\u064A\\u066E-\\u06D5\\u06E5-\\u06FF\\u0750-\\u077F\\u08A0-\\u08BD\\uFB50-\\uFBB1\\uFBD3-\\uFD3D\\uFD50-\\uFDC7\\uFDF0-\\uFDFB\\uFE70-\\uFEFC\\U0001EE00-\\U0001EEBB\\u0D80-\\u0DFF\\u0900-\\u097F\\u0C80-\\u0CFF\\u0B80-\\u0BFF\\u0C00-\\u0C7F\\uAC00-\\uD7AF\\u1100-\\u11FF\\u3040-\\u309F\\u30A0-\\u30FFー\\u4E00-\\u62FF\\u6300-\\u77FF\\u7800-\\u8CFF\\u8D00-\\u9FFF\\u3400-\\u4DBF\\U00020000-\\U000215FF\\U00021600-\\U000230FF\\U00023100-\\U000245FF\\U00024600-\\U000260FF\\U00026100-\\U000275FF\\U00027600-\\U000290FF\\U00029100-\\U0002A6DF\\U0002A700-\\U0002B73F\\U0002B740-\\U0002B81F\\U0002B820-\\U0002CEAF\\U0002CEB0-\\U0002EBEF\\u2E80-\\u2EFF\\u2F00-\\u2FDF\\u2FF0-\\u2FFF\\u3000-\\u303F\\u31C0-\\u31EF\\u3200-\\u32FF\\u3300-\\u33FF\\uF900-\\uFAFF\\uFE30-\\uFE4F\\U0001F200-\\U0001F2FF\\U0002F800-\\U0002FA1F])\\.$'

# Separate native callback wrapper; Tokenizer fields/layout stay unchanged.
cdef object _rmcp_single_suffix_match = re.compile("[" + re.escape(",:;!?()[]{}<>_#*&") + "]$").match

cdef class RMCPSuffixSearch:
    # Preserve ordinary Python attribute publication/deletion and weakrefs.
    cdef dict __dict__
    cdef object __weakref__
    cdef object _rmcp_english_pattern

    def __init__(self, pattern):
        self.pattern = pattern
        self._search = re.compile(pattern).search
        # Exact immutable string binding only; retain no extra callback/Pattern.
        self._rmcp_english_pattern = pattern if (type(pattern) is str and
            pattern == _rmcp_default_english_suffix_pattern) else None

    def __getattr__(self, name):
        # Match the original private Python wrapper's missing-attribute contract.
        raise AttributeError(f"'_SuffixSearch' object has no attribute '{name}'", name=name, obj=self)

    def search(self, text):
        # Keep Python object indexing semantics for string subclasses.
        if text and text[-1] in ",:;!?()[]{}<>_#*&":
            return _rmcp_single_suffix_match(text, len(text) - 1)
        return self._search(text)

cdef object _rmcp_suffix_function = RMCPSuffixSearch.search

cdef extern from *:
    """
    static int rmcp_suffix_callback(PyObject *callback, PyObject *function,
                                    PyObject *owner_type) {
        return PyMethod_Check(callback) &&
               PyMethod_GET_FUNCTION(callback) == function &&
               Py_IS_TYPE(PyMethod_GET_SELF(callback), (PyTypeObject *)owner_type);
    }
    static int rmcp_single_ascii_suffix(PyObject *text) {
        if (!PyUnicode_CheckExact(text)) return 0;
        Py_ssize_t length = PyUnicode_GET_LENGTH(text);
        if (!length) return 0;
        Py_UCS4 last = PyUnicode_READ_CHAR(text, length - 1);
        switch (last) {
            case ',': case ':': case ';': case '!': case '?':
            case '(': case ')': case '[': case ']': case '{': case '}':
            case '<': case '>': case '_': case '#': case '*': case '&':
                return 1;
            default: return 0;
        }
    }
    static Py_ssize_t rmcp_ascii_english_suffix(PyObject *text) {
        if (!PyUnicode_CheckExact(text) || !PyUnicode_IS_ASCII(text)) return -1;
        const unsigned char *p = PyUnicode_1BYTE_DATA(text);
        Py_ssize_t size = PyUnicode_GET_LENGTH(text);
        Py_ssize_t newline = size && p[size - 1] == 10;
        Py_ssize_t n = size - newline;
        if (!n) return 0;
        if (p[n - 1] == 46) {
            Py_ssize_t start = n - 1;
            while (start && p[start - 1] == 46) --start;
            if (n - start >= 2) return n - start;
        }
        switch (p[n - 1]) {
            case 33:
                if ((n >= 1)) return 1;
                return 0;
            case 34:
                if ((n >= 1)) return 1;
                return 0;
            case 35:
                if ((n >= 1)) return 1;
                return 0;
            case 36:
                if ((n >= 4) && (((p[n - 4] < 64 ? (0x03ff000000000000ULL >> p[n - 4]) : (0x0000000000000000ULL >> (p[n - 4] - 64))) & 1)) && (p[n - 3] == 85) && (p[n - 2] == 83)) return 3;
                if ((n >= 3) && (((p[n - 3] < 64 ? (0x03ff000000000000ULL >> p[n - 3]) : (0x0000000000000000ULL >> (p[n - 3] - 64))) & 1)) && (p[n - 2] == 67)) return 2;
                if ((n >= 3) && (((p[n - 3] < 64 ? (0x03ff000000000000ULL >> p[n - 3]) : (0x0000000000000000ULL >> (p[n - 3] - 64))) & 1)) && (p[n - 2] == 65)) return 2;
                if ((n >= 2) && (((p[n - 2] < 64 ? (0x03ff000000000000ULL >> p[n - 2]) : (0x0000000000000000ULL >> (p[n - 2] - 64))) & 1))) return 1;
                return 0;
            case 37:
                if ((n >= 2) && (((p[n - 2] < 64 ? (0x03ff000000000000ULL >> p[n - 2]) : (0x0000000000000000ULL >> (p[n - 2] - 64))) & 1))) return 1;
                return 0;
            case 38:
                if ((n >= 1)) return 1;
                return 0;
            case 39:
                if ((n >= 1)) return 1;
                return 0;
            case 40:
                if ((n >= 1)) return 1;
                return 0;
            case 41:
                if ((n >= 1)) return 1;
                return 0;
            case 42:
                if ((n >= 1)) return 1;
                return 0;
            case 43:
                if ((n >= 2) && (((p[n - 2] < 64 ? (0x03ff000000000000ULL >> p[n - 2]) : (0x0000000000000000ULL >> (p[n - 2] - 64))) & 1))) return 1;
                return 0;
            case 44:
                if ((n >= 1)) return 1;
                return 0;
            case 46:
                if ((n >= 2) && (((p[n - 2] < 64 ? (0xdfff3fee00000000ULL >> p[n - 2]) : (0x3fffffffa8000000ULL >> (p[n - 2] - 64))) & 1))) return 1;
                if ((n >= 3) && (((p[n - 3] < 64 ? (0x0000000000000000ULL >> p[n - 3]) : (0x0000000007fffffeULL >> (p[n - 3] - 64))) & 1)) && (((p[n - 2] < 64 ? (0x0000000000000000ULL >> p[n - 2]) : (0x0000000007fffffeULL >> (p[n - 2] - 64))) & 1))) return 1;
                return 0;
            case 58:
                if ((n >= 1)) return 1;
                return 0;
            case 59:
                if ((n >= 1)) return 1;
                return 0;
            case 60:
                if ((n >= 1)) return 1;
                return 0;
            case 62:
                if ((n >= 1)) return 1;
                return 0;
            case 63:
                if ((n >= 1)) return 1;
                return 0;
            case 66:
                if ((n >= 3) && (((p[n - 3] < 64 ? (0x03ff000000000000ULL >> p[n - 3]) : (0x0000000000000000ULL >> (p[n - 3] - 64))) & 1)) && (p[n - 2] == 77)) return 2;
                if ((n >= 3) && (((p[n - 3] < 64 ? (0x03ff000000000000ULL >> p[n - 3]) : (0x0000000000000000ULL >> (p[n - 3] - 64))) & 1)) && (p[n - 2] == 75)) return 2;
                if ((n >= 3) && (((p[n - 3] < 64 ? (0x03ff000000000000ULL >> p[n - 3]) : (0x0000000000000000ULL >> (p[n - 3] - 64))) & 1)) && (p[n - 2] == 71)) return 2;
                if ((n >= 3) && (((p[n - 3] < 64 ? (0x03ff000000000000ULL >> p[n - 3]) : (0x0000000000000000ULL >> (p[n - 3] - 64))) & 1)) && (p[n - 2] == 84)) return 2;
                return 0;
            case 71:
                if ((n >= 2) && (((p[n - 2] < 64 ? (0x03ff000000000000ULL >> p[n - 2]) : (0x0000000000000000ULL >> (p[n - 2] - 64))) & 1))) return 1;
                return 0;
            case 75:
                if ((n >= 2) && (((p[n - 2] < 64 ? (0x03ff000000000000ULL >> p[n - 2]) : (0x0000000000000000ULL >> (p[n - 2] - 64))) & 1))) return 1;
                return 0;
            case 77:
                if ((n >= 2) && (((p[n - 2] < 64 ? (0x03ff000000000000ULL >> p[n - 2]) : (0x0000000000000000ULL >> (p[n - 2] - 64))) & 1))) return 1;
                return 0;
            case 83:
                if ((n >= 2) && (p[n - 2] == 39)) return 2;
                return 0;
            case 84:
                if ((n >= 2) && (((p[n - 2] < 64 ? (0x03ff000000000000ULL >> p[n - 2]) : (0x0000000000000000ULL >> (p[n - 2] - 64))) & 1))) return 1;
                return 0;
            case 91:
                if ((n >= 1)) return 1;
                return 0;
            case 93:
                if ((n >= 1)) return 1;
                return 0;
            case 95:
                if ((n >= 1)) return 1;
                return 0;
            case 96:
                if ((n >= 1)) return 1;
                return 0;
            case 97:
                if ((n >= 4) && (((p[n - 4] < 64 ? (0x03ff000000000000ULL >> p[n - 4]) : (0x0000000000000000ULL >> (p[n - 4] - 64))) & 1)) && (p[n - 3] == 104) && (p[n - 2] == 80)) return 3;
                if ((n >= 3) && (((p[n - 3] < 64 ? (0x03ff000000000000ULL >> p[n - 3]) : (0x0000000000000000ULL >> (p[n - 3] - 64))) & 1)) && (p[n - 2] == 104)) return 2;
                if ((n >= 3) && (((p[n - 3] < 64 ? (0x03ff000000000000ULL >> p[n - 3]) : (0x0000000000000000ULL >> (p[n - 3] - 64))) & 1)) && (p[n - 2] == 80)) return 2;
                return 0;
            case 98:
                if ((n >= 3) && (((p[n - 3] < 64 ? (0x03ff000000000000ULL >> p[n - 3]) : (0x0000000000000000ULL >> (p[n - 3] - 64))) & 1)) && (p[n - 2] == 108)) return 2;
                if ((n >= 3) && (((p[n - 3] < 64 ? (0x03ff000000000000ULL >> p[n - 3]) : (0x0000000000000000ULL >> (p[n - 3] - 64))) & 1)) && (p[n - 2] == 109)) return 2;
                if ((n >= 3) && (((p[n - 3] < 64 ? (0x03ff000000000000ULL >> p[n - 3]) : (0x0000000000000000ULL >> (p[n - 3] - 64))) & 1)) && (p[n - 2] == 107)) return 2;
                if ((n >= 3) && (((p[n - 3] < 64 ? (0x03ff000000000000ULL >> p[n - 3]) : (0x0000000000000000ULL >> (p[n - 3] - 64))) & 1)) && (p[n - 2] == 103)) return 2;
                if ((n >= 3) && (((p[n - 3] < 64 ? (0x03ff000000000000ULL >> p[n - 3]) : (0x0000000000000000ULL >> (p[n - 3] - 64))) & 1)) && (p[n - 2] == 116)) return 2;
                return 0;
            case 100:
                if ((n >= 3) && (((p[n - 3] < 64 ? (0x03ff000000000000ULL >> p[n - 3]) : (0x0000000000000000ULL >> (p[n - 3] - 64))) & 1)) && (p[n - 2] == 121)) return 2;
                return 0;
            case 103:
                if ((n >= 3) && (((p[n - 3] < 64 ? (0x03ff000000000000ULL >> p[n - 3]) : (0x0000000000000000ULL >> (p[n - 3] - 64))) & 1)) && (p[n - 2] == 107)) return 2;
                if ((n >= 3) && (((p[n - 3] < 64 ? (0x03ff000000000000ULL >> p[n - 3]) : (0x0000000000000000ULL >> (p[n - 3] - 64))) & 1)) && (p[n - 2] == 109)) return 2;
                if ((n >= 2) && (((p[n - 2] < 64 ? (0x03ff000000000000ULL >> p[n - 2]) : (0x0000000000000000ULL >> (p[n - 2] - 64))) & 1))) return 1;
                return 0;
            case 104:
                if ((n >= 5) && (((p[n - 5] < 64 ? (0x03ff000000000000ULL >> p[n - 5]) : (0x0000000000000000ULL >> (p[n - 5] - 64))) & 1)) && (p[n - 4] == 107) && (p[n - 3] == 109) && (p[n - 2] == 47)) return 4;
                if ((n >= 4) && (((p[n - 4] < 64 ? (0x03ff000000000000ULL >> p[n - 4]) : (0x0000000000000000ULL >> (p[n - 4] - 64))) & 1)) && (p[n - 3] == 107) && (p[n - 2] == 109)) return 3;
                if ((n >= 4) && (((p[n - 4] < 64 ? (0x03ff000000000000ULL >> p[n - 4]) : (0x0000000000000000ULL >> (p[n - 4] - 64))) & 1)) && (p[n - 3] == 109) && (p[n - 2] == 112)) return 3;
                return 0;
            case 109:
                if ((n >= 3) && (((p[n - 3] < 64 ? (0x03ff000000000000ULL >> p[n - 3]) : (0x0000000000000000ULL >> (p[n - 3] - 64))) & 1)) && (p[n - 2] == 107)) return 2;
                if ((n >= 3) && (((p[n - 3] < 64 ? (0x03ff000000000000ULL >> p[n - 3]) : (0x0000000000000000ULL >> (p[n - 3] - 64))) & 1)) && (p[n - 2] == 100)) return 2;
                if ((n >= 3) && (((p[n - 3] < 64 ? (0x03ff000000000000ULL >> p[n - 3]) : (0x0000000000000000ULL >> (p[n - 3] - 64))) & 1)) && (p[n - 2] == 99)) return 2;
                if ((n >= 3) && (((p[n - 3] < 64 ? (0x03ff000000000000ULL >> p[n - 3]) : (0x0000000000000000ULL >> (p[n - 3] - 64))) & 1)) && (p[n - 2] == 109)) return 2;
                if ((n >= 3) && (((p[n - 3] < 64 ? (0x03ff000000000000ULL >> p[n - 3]) : (0x0000000000000000ULL >> (p[n - 3] - 64))) & 1)) && (p[n - 2] == 110)) return 2;
                if ((n >= 2) && (((p[n - 2] < 64 ? (0x03ff000000000000ULL >> p[n - 2]) : (0x0000000000000000ULL >> (p[n - 2] - 64))) & 1))) return 1;
                return 0;
            case 110:
                if ((n >= 3) && (((p[n - 3] < 64 ? (0x03ff000000000000ULL >> p[n - 3]) : (0x0000000000000000ULL >> (p[n - 3] - 64))) & 1)) && (p[n - 2] == 105)) return 2;
                return 0;
            case 114:
                if ((n >= 5) && (((p[n - 5] < 64 ? (0x03ff000000000000ULL >> p[n - 5]) : (0x0000000000000000ULL >> (p[n - 5] - 64))) & 1)) && (p[n - 4] == 109) && (p[n - 3] == 98) && (p[n - 2] == 97)) return 4;
                return 0;
            case 115:
                if ((n >= 4) && (((p[n - 4] < 64 ? (0x03ff000000000000ULL >> p[n - 4]) : (0x0000000000000000ULL >> (p[n - 4] - 64))) & 1)) && (p[n - 3] == 109) && (p[n - 2] == 47)) return 3;
                if ((n >= 2) && (p[n - 2] == 39)) return 2;
                return 0;
            case 116:
                if ((n >= 3) && (((p[n - 3] < 64 ? (0x03ff000000000000ULL >> p[n - 3]) : (0x0000000000000000ULL >> (p[n - 3] - 64))) & 1)) && (p[n - 2] == 102)) return 2;
                if ((n >= 2) && (((p[n - 2] < 64 ? (0x03ff000000000000ULL >> p[n - 2]) : (0x0000000000000000ULL >> (p[n - 2] - 64))) & 1))) return 1;
                return 0;
            case 122:
                if ((n >= 3) && (((p[n - 3] < 64 ? (0x03ff000000000000ULL >> p[n - 3]) : (0x0000000000000000ULL >> (p[n - 3] - 64))) & 1)) && (p[n - 2] == 111)) return 2;
                return 0;
            case 123:
                if ((n >= 1)) return 1;
                return 0;
            case 125:
                if ((n >= 1)) return 1;
                return 0;
            default: return 0;
        }
    }
    static PyCFunction rmcp_suffix_regex_search_function = NULL;
    static void rmcp_set_suffix_regex_search_function(PyObject *callback) {
        rmcp_suffix_regex_search_function = PyCFunction_GET_FUNCTION(callback);
    }
    static int rmcp_exact_default_suffix(PyObject *callback, PyObject *pattern_type,
                                          PyObject *expected_pattern) {
        if (expected_pattern == Py_None || !PyCFunction_Check(callback) ||
            PyCFunction_GET_FUNCTION(callback) != rmcp_suffix_regex_search_function)
            return 0;
        PyObject *owner = PyCFunction_GET_SELF(callback);
        if (owner == NULL || !Py_IS_TYPE(owner, (PyTypeObject *)pattern_type)) return 0;
        PyObject *pattern = PyObject_GetAttrString(owner, "pattern");
        if (pattern == NULL) return -1;
        int equal = pattern == expected_pattern;
        Py_DECREF(pattern);
        if (!equal) return 0;
        PyObject *flags = PyObject_GetAttrString(owner, "flags");
        if (flags == NULL) return -1;
        long value = PyLong_AsLong(flags);
        Py_DECREF(flags);
        if (value == -1 && PyErr_Occurred()) return -1;
        return value == 32;
    }
    """
    bint rmcp_suffix_callback(object callback, object function, object owner_type)
    bint rmcp_single_ascii_suffix(object text)

cdef extern from *:
    void rmcp_set_suffix_regex_search_function(object callback)
    int rmcp_exact_default_suffix(object callback, object pattern_type, object expected_pattern) except -1
    Py_ssize_t rmcp_ascii_english_suffix(object text)

rmcp_set_suffix_regex_search_function(re.compile('').search)

cdef inline bint _rmcp_has_default_suffix_search(RMCPSuffixSearch owner) except -1:
    # Confine the temporary owner reference to classification. Custom fallbacks
    # must release it before invoking an arbitrary callback/Match-like result.
    return rmcp_exact_default_suffix(owner._search, _RegexPattern, owner._rmcp_english_pattern)

from weakref import ref as _rmcp_prefix_weakref

cdef str _rmcp_english_prefix_pattern = '^§|^%|^=|^—|^–|^\\+(?![0-9])|^…|^……|^,|^:|^;|^\\!|^\\?|^¿|^؟|^¡|^\\(|^\\)|^\\[|^\\]|^\\{|^\\}|^<|^>|^_|^#|^\\*|^&|^。|^？|^！|^，|^、|^；|^：|^～|^·|^।|^،|^۔|^؛|^٪|^\\.\\.+|^…|^\\\'|^"|^”|^“|^`|^‘|^´|^’|^‚|^,|^„|^»|^«|^「|^」|^『|^』|^（|^）|^〔|^〕|^【|^】|^《|^》|^〈|^〉|^〈|^〉|^⟦|^⟧|^\\$|^£|^€|^¥|^฿|^US\\$|^C\\$|^A\\$|^₽|^﷼|^₴|^₠|^₡|^₢|^₣|^₤|^₥|^₦|^₧|^₨|^₩|^₪|^₫|^€|^₭|^₮|^₯|^₰|^₱|^₲|^₳|^₴|^₵|^₶|^₷|^₸|^₹|^₺|^₻|^₼|^₽|^₾|^₿|^[\\u00A6\\u00A9\\u00AE\\u00B0\\u0482\\u058D\\u058E\\u060E\\u060F\\u06DE\\u06E9\\u06FD\\u06FE\\u07F6\\u09FA\\u0B70\\u0BF3-\\u0BF8\\u0BFA\\u0C7F\\u0D4F\\u0D79\\u0F01-\\u0F03\\u0F13\\u0F15-\\u0F17\\u0F1A-\\u0F1F\\u0F34\\u0F36\\u0F38\\u0FBE-\\u0FC5\\u0FC7-\\u0FCC\\u0FCE\\u0FCF\\u0FD5-\\u0FD8\\u109E\\u109F\\u1390-\\u1399\\u1940\\u19DE-\\u19FF\\u1B61-\\u1B6A\\u1B74-\\u1B7C\\u2100\\u2101\\u2103-\\u2106\\u2108\\u2109\\u2114\\u2116\\u2117\\u211E-\\u2123\\u2125\\u2127\\u2129\\u212E\\u213A\\u213B\\u214A\\u214C\\u214D\\u214F\\u218A\\u218B\\u2195-\\u2199\\u219C-\\u219F\\u21A1\\u21A2\\u21A4\\u21A5\\u21A7-\\u21AD\\u21AF-\\u21CD\\u21D0\\u21D1\\u21D3\\u21D5-\\u21F3\\u2300-\\u2307\\u230C-\\u231F\\u2322-\\u2328\\u232B-\\u237B\\u237D-\\u239A\\u23B4-\\u23DB\\u23E2-\\u2426\\u2440-\\u244A\\u249C-\\u24E9\\u2500-\\u25B6\\u25B8-\\u25C0\\u25C2-\\u25F7\\u2600-\\u266E\\u2670-\\u2767\\u2794-\\u27BF\\u2800-\\u28FF\\u2B00-\\u2B2F\\u2B45\\u2B46\\u2B4D-\\u2B73\\u2B76-\\u2B95\\u2B98-\\u2BC8\\u2BCA-\\u2BFE\\u2CE5-\\u2CEA\\u2E80-\\u2E99\\u2E9B-\\u2EF3\\u2F00-\\u2FD5\\u2FF0-\\u2FFB\\u3004\\u3012\\u3013\\u3020\\u3036\\u3037\\u303E\\u303F\\u3190\\u3191\\u3196-\\u319F\\u31C0-\\u31E3\\u3200-\\u321E\\u322A-\\u3247\\u3250\\u3260-\\u327F\\u328A-\\u32B0\\u32C0-\\u32FE\\u3300-\\u33FF\\u4DC0-\\u4DFF\\uA490-\\uA4C6\\uA828-\\uA82B\\uA836\\uA837\\uA839\\uAA77-\\uAA79\\uFDFD\\uFFE4\\uFFE8\\uFFED\\uFFEE\\uFFFC\\uFFFD\\U00010137-\\U0001013F\\U00010179-\\U00010189\\U0001018C-\\U0001018E\\U00010190-\\U0001019B\\U000101A0\\U000101D0-\\U000101FC\\U00010877\\U00010878\\U00010AC8\\U0001173F\\U00016B3C-\\U00016B3F\\U00016B45\\U0001BC9C\\U0001D000-\\U0001D0F5\\U0001D100-\\U0001D126\\U0001D129-\\U0001D164\\U0001D16A-\\U0001D16C\\U0001D183\\U0001D184\\U0001D18C-\\U0001D1A9\\U0001D1AE-\\U0001D1E8\\U0001D200-\\U0001D241\\U0001D245\\U0001D300-\\U0001D356\\U0001D800-\\U0001D9FF\\U0001DA37-\\U0001DA3A\\U0001DA6D-\\U0001DA74\\U0001DA76-\\U0001DA83\\U0001DA85\\U0001DA86\\U0001ECAC\\U0001F000-\\U0001F02B\\U0001F030-\\U0001F093\\U0001F0A0-\\U0001F0AE\\U0001F0B1-\\U0001F0BF\\U0001F0C1-\\U0001F0CF\\U0001F0D1-\\U0001F0F5\\U0001F110-\\U0001F16B\\U0001F170-\\U0001F1AC\\U0001F1E6-\\U0001F202\\U0001F210-\\U0001F23B\\U0001F240-\\U0001F248\\U0001F250\\U0001F251\\U0001F260-\\U0001F265\\U0001F300-\\U0001F3FA\\U0001F400-\\U0001F6D4\\U0001F6E0-\\U0001F6EC\\U0001F6F0-\\U0001F6F9\\U0001F700-\\U0001F773\\U0001F780-\\U0001F7D8\\U0001F800-\\U0001F80B\\U0001F810-\\U0001F847\\U0001F850-\\U0001F859\\U0001F860-\\U0001F887\\U0001F890-\\U0001F8AD\\U0001F900-\\U0001F90B\\U0001F910-\\U0001F93E\\U0001F940-\\U0001F970\\U0001F973-\\U0001F976\\U0001F97A\\U0001F97C-\\U0001F9A2\\U0001F9B0-\\U0001F9B9\\U0001F9C0-\\U0001F9C2\\U0001F9D0-\\U0001F9FF\\U0001FA60-\\U0001FA6D]'
cdef object _rmcp_prefix_callback_ref = None

cdef extern from *:
    r"""
    static PyCFunction rmcp_prefix_regex_search_function = NULL;
    static void rmcp_set_prefix_regex_search_function(PyObject *callback) {
        rmcp_prefix_regex_search_function = PyCFunction_GET_FUNCTION(callback);
    }
    static int rmcp_exact_english_prefix(PyObject *callback, PyObject *pattern_type,
                                         PyObject *expected_pattern) {
        if (!PyCFunction_Check(callback) ||
            PyCFunction_GET_FUNCTION(callback) != rmcp_prefix_regex_search_function)
            return 0;
        PyObject *owner = PyCFunction_GET_SELF(callback);
        if (owner == NULL || !Py_IS_TYPE(owner, (PyTypeObject *)pattern_type)) return 0;
        PyObject *pattern = PyObject_GetAttrString(owner, "pattern");
        if (pattern == NULL) return -1;
        int equal = PyUnicode_CheckExact(pattern) &&
            PyUnicode_Compare(pattern, expected_pattern) == 0;
        Py_DECREF(pattern);
        if (!equal) return 0;
        PyObject *flags = PyObject_GetAttrString(owner, "flags");
        if (flags == NULL) return -1;
        long value = PyLong_AsLong(flags);
        Py_DECREF(flags);
        if (value == -1 && PyErr_Occurred()) return -1;
        return value == 32;
    }
    static const unsigned char rmcp_prefix_ascii[128] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0};
    static const unsigned int rmcp_prefix_ranges[][2] = {{161,161},{163,163},{165,167},{169,169},{171,171},{174,174},{176,176},{180,180},{183,183},{187,187},{191,191},{1154,1154},{1421,1422},{1548,1548},{1550,1551},{1563,1563},{1567,1567},{1642,1642},{1748,1748},{1758,1758},{1769,1769},{1789,1790},{2038,2038},{2404,2404},{2554,2554},{2928,2928},{3059,3064},{3066,3066},{3199,3199},{3407,3407},{3449,3449},{3647,3647},{3841,3843},{3859,3859},{3861,3863},{3866,3871},{3892,3892},{3894,3894},{3896,3896},{4030,4037},{4039,4044},{4046,4047},{4053,4056},{4254,4255},{5008,5017},{6464,6464},{6622,6655},{7009,7018},{7028,7036},{8211,8212},{8216,8218},{8220,8222},{8230,8230},{8352,8383},{8448,8449},{8451,8454},{8456,8457},{8468,8468},{8470,8471},{8478,8483},{8485,8485},{8487,8487},{8489,8489},{8494,8494},{8506,8507},{8522,8522},{8524,8525},{8527,8527},{8586,8587},{8597,8601},{8604,8607},{8609,8610},{8612,8613},{8615,8621},{8623,8653},{8656,8657},{8659,8659},{8661,8691},{8960,8967},{8972,8991},{8994,9083},{9085,9114},{9140,9179},{9186,9254},{9280,9290},{9372,9449},{9472,9654},{9656,9664},{9666,9719},{9728,9838},{9840,10087},{10132,10175},{10214,10215},{10240,10495},{11008,11055},{11077,11078},{11085,11123},{11126,11157},{11160,11208},{11210,11262},{11493,11498},{11904,11929},{11931,12019},{12032,12245},{12272,12283},{12289,12290},{12292,12292},{12296,12309},{12320,12320},{12342,12343},{12350,12351},{12688,12689},{12694,12703},{12736,12771},{12800,12830},{12842,12871},{12880,12880},{12896,12927},{12938,12976},{12992,13054},{13056,13311},{19904,19967},{42128,42182},{43048,43051},{43062,43063},{43065,43065},{43639,43641},{65020,65021},{65281,65281},{65288,65289},{65292,65292},{65306,65307},{65311,65311},{65374,65374},{65508,65508},{65512,65512},{65517,65518},{65532,65533},{65847,65855},{65913,65929},{65932,65934},{65936,65947},{65952,65952},{66000,66044},{67703,67704},{68296,68296},{71487,71487},{92988,92991},{92997,92997},{113820,113820},{118784,119029},{119040,119078},{119081,119140},{119146,119148},{119171,119172},{119180,119209},{119214,119272},{119296,119361},{119365,119365},{119552,119638},{120832,121343},{121399,121402},{121453,121460},{121462,121475},{121477,121478},{126124,126124},{126976,127019},{127024,127123},{127136,127150},{127153,127167},{127169,127183},{127185,127221},{127248,127339},{127344,127404},{127462,127490},{127504,127547},{127552,127560},{127568,127569},{127584,127589},{127744,127994},{128000,128724},{128736,128748},{128752,128761},{128768,128883},{128896,128984},{129024,129035},{129040,129095},{129104,129113},{129120,129159},{129168,129197},{129280,129291},{129296,129342},{129344,129392},{129395,129398},{129402,129402},{129404,129442},{129456,129465},{129472,129474},{129488,129535},{129632,129645}};
    static Py_ssize_t rmcp_known_prefix_length(PyObject *text, PyObject *callback, PyObject *reference) {
        if (!PyUnicode_CheckExact(text) || reference == Py_None) return -1;
        PyObject *known_callback = PyWeakref_GetObject(reference);
        if (known_callback == Py_None || known_callback != callback) return -1;
        Py_ssize_t n = PyUnicode_GET_LENGTH(text);
        if (n == 0) return 0;
        Py_UCS4 c = PyUnicode_READ_CHAR(text, 0);
        if (c == '.') {
            if (n < 2 || PyUnicode_READ_CHAR(text, 1) != '.') return 0;
            Py_ssize_t end = 2;
            while (end < n && PyUnicode_READ_CHAR(text, end) == '.') ++end;
            return end;
        }
        if (c == '+') return (n > 1 && PyUnicode_READ_CHAR(text, 1) >= '0' && PyUnicode_READ_CHAR(text, 1) <= '9') ? 0 : 1;
        if (c == 'A' || c == 'C') return (n > 1 && PyUnicode_READ_CHAR(text, 1) == '$') ? 2 : 0;
        if (c == 'U') return (n > 2 && PyUnicode_READ_CHAR(text, 1) == 'S' && PyUnicode_READ_CHAR(text, 2) == '$') ? 3 : 0;
        if (c < 128) return rmcp_prefix_ascii[c];
        size_t lo = 0, hi = sizeof(rmcp_prefix_ranges)/sizeof(rmcp_prefix_ranges[0]);
        while (lo < hi) {
            size_t mid = lo + (hi-lo)/2;
            if (c < rmcp_prefix_ranges[mid][0]) hi=mid;
            else if (c > rmcp_prefix_ranges[mid][1]) lo=mid+1;
            else return 1;
        }
        return 0;
    }
    """
    void rmcp_set_prefix_regex_search_function(object callback)
    int rmcp_exact_english_prefix(object callback, object pattern_type, object expected_pattern) except -1
    Py_ssize_t rmcp_known_prefix_length(object text, object callback, object reference)

rmcp_set_prefix_regex_search_function(re.compile('').search)

cdef class Tokenizer:
    """Segment text, and create Doc objects with the discovered segment
    boundaries.

    DOCS: https://spacy.io/api/tokenizer
    """
    def __init__(self, Vocab vocab, rules=None, prefix_search=None,
                 suffix_search=None, infix_finditer=None, token_match=None,
                 url_match=None, faster_heuristics=True, max_cache_size=10000):
        """Create a `Tokenizer`, to create `Doc` objects given unicode text.

        vocab (Vocab): A storage container for lexical types.
        rules (dict): Exceptions and special-cases for the tokenizer.
        prefix_search (callable): A function matching the signature of
            `re.compile(string).search` to match prefixes.
        suffix_search (callable): A function matching the signature of
            `re.compile(string).search` to match suffixes.
        infix_finditer (callable): A function matching the signature of
            `re.compile(string).finditer` to find infixes.
        token_match (callable): A function matching the signature of
            `re.compile(string).match`, for matching strings to be
            recognized as tokens.
        url_match (callable): A function matching the signature of
            `re.compile(string).match`, for matching strings to be
            recognized as urls.
        faster_heuristics (bool): Whether to restrict the final
            Matcher-based pass for rules to those containing affixes or space.
            Defaults to True.
        max_cache_size (int): Maximum number of tokenization chunks to cache.

        EXAMPLE:
            >>> tokenizer = Tokenizer(nlp.vocab)

        DOCS: https://spacy.io/api/tokenizer#init
        """
        self.mem = Pool()
        self._cache = PreshMap()
        self._specials = PreshMap()
        self.token_match = token_match
        self.url_match = url_match
        self.prefix_search = prefix_search
        self.suffix_search = suffix_search
        self.infix_finditer = infix_finditer
        self.vocab = vocab
        self.faster_heuristics = faster_heuristics
        self._rules = {}
        self._special_matcher = PhraseMatcher(self.vocab)
        self._load_special_cases(rules)
        self.max_cache_size = max_cache_size

    @property
    def token_match(self):
        return self._token_match

    @token_match.setter
    def token_match(self, token_match):
        # Invalidate before replacing: old callback destruction can reenter.
        self._rmcp_ascii_word_match = False
        old_token_match = self._token_match
        self._token_match = token_match
        # Publish replacement before old destruction can reenter the tokenizer.
        old_token_match = None
        # Destruction may have assigned another callback; classify current state.
        self._rmcp_ascii_word_match = type(self) is Tokenizer and rmcp_exact_ascii_pattern(self._token_match, _RegexPattern)
        self._reload_special_cases()

    @property
    def url_match(self):
        return self._url_match

    @url_match.setter
    def url_match(self, url_match):
        self._url_match = url_match
        self._reload_special_cases()

    @property
    def prefix_search(self):
        return self._prefix_search

    @prefix_search.setter
    def prefix_search(self, prefix_search):
        global _rmcp_prefix_callback_ref
        self._prefix_search = prefix_search
        # Classification reads only exact immutable builtin Pattern fields.
        # No Tokenizer layout field or strong callback/Pattern reference is added.
        if type(self) is Tokenizer and rmcp_exact_english_prefix(self._prefix_search, _RegexPattern, _rmcp_english_prefix_pattern):
            _rmcp_prefix_callback_ref = _rmcp_prefix_weakref(self._prefix_search)
        self._reload_special_cases()

    @property
    def suffix_search(self):
        return self._suffix_search

    @suffix_search.setter
    def suffix_search(self, suffix_search):
        self._suffix_search = suffix_search
        self._reload_special_cases()

    @property
    def infix_finditer(self):
        return self._infix_finditer

    @infix_finditer.setter
    def infix_finditer(self, infix_finditer):
        self._infix_finditer = infix_finditer
        self._reload_special_cases()

    @property
    def rules(self):
        return self._rules

    @rules.setter
    def rules(self, rules):
        self._rules = {}
        self._flush_cache()
        self._flush_specials()
        self._cache = PreshMap()
        self._specials = PreshMap()
        self._load_special_cases(rules)

    @property
    def faster_heuristics(self):
        return bool(self._faster_heuristics)

    @faster_heuristics.setter
    def faster_heuristics(self, faster_heuristics):
        self._faster_heuristics = bool(faster_heuristics)
        self._reload_special_cases()

    def __reduce__(self):
        args = (self.vocab,
                self.rules,
                self.prefix_search,
                self.suffix_search,
                self.infix_finditer,
                self.token_match,
                self.url_match)
        return (self.__class__, args, None, None)

    def __call__(self, str string):
        """Tokenize a string.

        string (str): The string to tokenize.
        RETURNS (Doc): A container for linguistic annotations.

        DOCS: https://spacy.io/api/tokenizer#call
        """
        doc = self._tokenize_affixes(string, True)
        self._apply_special_cases(doc)
        return doc

    @cython.boundscheck(False)
    cdef Doc _tokenize_affixes(self, str string, bint with_special_cases):
        """Tokenize according to affix and token_match settings.

        string (str): The string to tokenize.
        RETURNS (Doc): A container for linguistic annotations.
        """
        if len(string) >= (2 ** 30):
            raise ValueError(Errors.E025.format(length=len(string)))
        cdef int length = len(string)
        cdef Doc doc = Doc(self.vocab)
        if length == 0:
            return doc
        cdef int i = 0
        cdef int start = 0
        cdef int has_special = 0
        cdef bint in_ws = string[0].isspace()
        cdef str span
        cdef int block_start = 1
        cdef unsigned block_width
        cdef uint64_t transitions
        if rmcp_exact_unicode(string):
            # Each character is classified once. Consume every transition in
            # the block, including runs crossing its boundary, before loading
            # the next block. The original subclass loop remains below.
            while block_start < length:
                block_width = 64 if length - block_start >= 64 else length - block_start
                transitions = rmcp_space_transition_block(string, block_start, block_width, in_ws)
                while transitions:
                    i = block_start + rmcp_first_set_bit(transitions)
                    transitions &= transitions - 1
                    if start < i:
                        span = string[start:i]
                        key = hash_string(span)
                        if not self._try_specials_and_cache(key, doc, &has_special, with_special_cases):
                            self._tokenize(doc, span, key, &has_special, with_special_cases)
                    if string[i] == ' ':
                        doc.c[doc.length - 1].spacy = True
                        start = i + 1
                    else:
                        start = i
                    in_ws = not in_ws
                block_start += block_width
            i = length
            if start < i:
                span = string[start:]
                key = hash_string(span)
                if not self._try_specials_and_cache(key, doc, &has_special, with_special_cases):
                    self._tokenize(doc, span, key, &has_special, with_special_cases)
                doc.c[doc.length - 1].spacy = string[-1] == " " and not in_ws
            return doc
        # The task here is much like string.split, but not quite
        # We find spans of whitespace and non-space characters, and ignore
        # spans that are exactly ' '. So, our sequences will all be separated
        # by either ' ' or nothing.
        for uc in string:
            if uc.isspace() != in_ws:
                if start < i:
                    # When we want to make this fast, get the data buffer once
                    # with PyUnicode_AS_DATA, and then maintain a start_byte
                    # and end_byte, so we can call hash64 directly. That way
                    # we don't have to create the slice when we hit the cache.
                    span = string[start:i]
                    key = hash_string(span)
                    if not self._try_specials_and_cache(key, doc, &has_special, with_special_cases):
                        self._tokenize(doc, span, key, &has_special, with_special_cases)
                if uc == ' ':
                    doc.c[doc.length - 1].spacy = True
                    start = i + 1
                else:
                    start = i
                in_ws = not in_ws
            i += 1
        if start < i:
            span = string[start:]
            key = hash_string(span)
            if not self._try_specials_and_cache(key, doc, &has_special, with_special_cases):
                self._tokenize(doc, span, key, &has_special, with_special_cases)
            doc.c[doc.length - 1].spacy = string[-1] == " " and not in_ws
        return doc

    def pipe(self, texts, batch_size=1000):
        """Tokenize a stream of texts.

        texts: A sequence of unicode texts.
        batch_size (int): Number of texts to accumulate in an internal buffer.
        Defaults to 1000.
        YIELDS (Doc): A sequence of Doc objects, in order.

        DOCS: https://spacy.io/api/tokenizer#pipe
        """
        for text in texts:
            yield self(text)

    def _flush_cache(self):
        self._reset_cache([key for key in self._cache])

    def _reset_cache(self, keys):
        for k in keys:
            cached = <_Cached*>self._cache.get(k)
            del self._cache[k]
            if cached is not NULL:
                self.mem.free(cached)

    def _flush_specials(self):
        self._special_matcher = PhraseMatcher(self.vocab)
        for k in self._specials:
            cached = <_Cached*>self._specials.get(k)
            del self._specials[k]
            if cached is not NULL:
                self.mem.free(cached)

    cdef int _apply_special_cases(self, Doc doc) except -1:
        """Retokenize doc according to special cases.

        doc (Doc): Document.
        """
        cdef int i
        cdef int max_length = 0
        cdef bint modify_in_place
        cdef Pool mem = Pool()
        cdef vector[SpanC] c_matches
        cdef vector[SpanC] c_filtered
        cdef int offset
        cdef int modified_doc_length
        # Find matches for special cases
        self._special_matcher.find_matches(doc, 0, doc.length, &c_matches)
        # Skip processing if no matches
        if c_matches.size() == 0:
            return True
        self._filter_special_spans(c_matches, c_filtered, doc.length)
        # Put span info in span.start-indexed dict and calculate maximum
        # intermediate document size
        (span_data, max_length, modify_in_place) = self._prepare_special_spans(doc, c_filtered)
        # If modifications never increase doc length, can modify in place
        if modify_in_place:
            tokens = doc.c
        # Otherwise create a separate array to store modified tokens
        else:
            assert max_length > 0
            tokens = <TokenC*>mem.alloc(max_length, sizeof(TokenC))
        # Modify tokenization according to filtered special cases
        offset = self._retokenize_special_spans(doc, tokens, span_data)
        # Allocate more memory for doc if needed
        modified_doc_length = doc.length + offset
        while modified_doc_length >= doc.max_length:
            doc._realloc(doc.max_length * 2)
        # If not modified in place, copy tokens back to doc
        if not modify_in_place:
            memcpy(doc.c, tokens, max_length * sizeof(TokenC))
        for i in range(doc.length + offset, doc.length):
            memset(&doc.c[i], 0, sizeof(TokenC))
            doc.c[i].lex = &EMPTY_LEXEME
        doc.length = doc.length + offset
        return True

    cdef void _filter_special_spans(self, vector[SpanC] &original, vector[SpanC] &filtered, int doc_len) nogil:

        cdef int seen_i
        cdef SpanC span
        cdef stdset[int] seen_tokens
        stdsort(original.begin(), original.end(), len_start_cmp)
        cdef int orig_i = original.size() - 1
        while orig_i >= 0:
            span = original[orig_i]
            if not seen_tokens.count(span.start) and not seen_tokens.count(span.end - 1):
                filtered.push_back(span)
            for seen_i in range(span.start, span.end):
                seen_tokens.insert(seen_i)
            orig_i -= 1
        stdsort(filtered.begin(), filtered.end(), start_cmp)

    cdef object _prepare_special_spans(self, Doc doc, vector[SpanC] &filtered):
        spans = [doc[match.start:match.end] for match in filtered]
        cdef bint modify_in_place = True
        cdef int curr_length = doc.length
        cdef int max_length = 0
        cdef int span_length_diff = 0
        span_data = {}
        for span in spans:
            rule = self._rules.get(span.text, None)
            span_length_diff = 0
            if rule:
                span_length_diff = len(rule) - (span.end - span.start)
            if span_length_diff > 0:
                modify_in_place = False
            curr_length += span_length_diff
            if curr_length > max_length:
                max_length = curr_length
            span_data[span.start] = (span.text, span.start, span.end, span_length_diff)
        return (span_data, max_length, modify_in_place)

    cdef int _retokenize_special_spans(self, Doc doc, TokenC* tokens, object span_data):
        cdef int i = 0
        cdef int j = 0
        cdef int offset = 0
        cdef _Cached* cached
        cdef int idx_offset = 0
        cdef int orig_final_spacy
        cdef int orig_idx
        cdef int span_start
        cdef int span_end
        while i < doc.length:
            if i not in span_data:
                tokens[i + offset] = doc.c[i]
                i += 1
            else:
                span = span_data[i]
                span_start = span[1]
                span_end = span[2]
                cached = <_Cached*>self._specials.get(hash_string(span[0]))
                if cached == NULL:
                    # Copy original tokens if no rule found
                    for j in range(span_end - span_start):
                        tokens[i + offset + j] = doc.c[i + j]
                    i += span_end - span_start
                else:
                    # Copy special case tokens into doc and adjust token and
                    # character offsets
                    idx_offset = 0
                    orig_final_spacy = doc.c[span_end - 1].spacy
                    orig_idx = doc.c[i].idx
                    for j in range(cached.length):
                        tokens[i + offset + j] = cached.data.tokens[j]
                        tokens[i + offset + j].idx = orig_idx + idx_offset
                        idx_offset += cached.data.tokens[j].lex.length
                        if cached.data.tokens[j].spacy:
                            idx_offset += 1
                    tokens[i + offset + cached.length - 1].spacy = orig_final_spacy
                    i += span_end - span_start
                    offset += span[3]
        return offset

    cdef int _try_specials_and_cache(self, hash_t key, Doc tokens, int* has_special, bint with_special_cases) except -1:
        cdef bint specials_hit = 0
        cdef bint cache_hit = 0
        cdef int i
        if with_special_cases:
            cached = <_Cached*>self._specials.get(key)
            if cached == NULL:
                specials_hit = False
            else:
                for i in range(cached.length):
                    tokens.push_back(&cached.data.tokens[i], False)
                has_special[0] = 1
                specials_hit = True
        if not specials_hit:
            cached = <_Cached*>self._cache.get(key)
            if cached == NULL:
                cache_hit = False
            else:
                if cached.is_lex:
                    for i in range(cached.length):
                        tokens.push_back(cached.data.lexemes[i], False)
                else:
                    for i in range(cached.length):
                        tokens.push_back(&cached.data.tokens[i], False)
                cache_hit = True
        if not specials_hit and not cache_hit:
            return False
        return True

    cdef int _tokenize(self, Doc tokens, str span, hash_t orig_key, int* has_special, bint with_special_cases) except -1:
        cdef vector[LexemeC*] prefixes
        cdef vector[LexemeC*] suffixes
        cdef int orig_size
        cdef const LexemeC* word_lexeme
        orig_size = tokens.length
        if type(self) is Tokenizer and type(self.vocab) is Vocab and self._rmcp_ascii_word_match and rmcp_ascii_word_text(span):
            # The caller already missed specials/cache. This exact builtin word
            # path breaks _split_affixes without a callback, then repeats the
            # same hash/cache miss in _attach_tokens before appending this lexeme.
            # orig_key already hashes this unchanged word. Existing vocabulary
            # entries need no second StringStore hash or membership lookup.
            # A missing/reserved-symbol key keeps the ordinary creation path.
            word_lexeme = <const LexemeC*>self.vocab._by_orth.get(orig_key)
            if word_lexeme == NULL:
                word_lexeme = self.vocab.get(tokens.mem, span)
            tokens.push_back(word_lexeme, False)
        else:
            span = self._split_affixes(tokens.mem, span, &prefixes, &suffixes,
                                       has_special, with_special_cases)
            self._attach_tokens(tokens, span, &prefixes, &suffixes, has_special,
                                with_special_cases)
        # Exact base instances cannot save while the document's special flag is set.
        # Keep the original call path for native subclasses overriding _save_cached.
        if (type(self) is not Tokenizer or type(self.vocab) is not Vocab or not has_special[0]) and len(self._cache) < self.max_cache_size:
            self._save_cached(&tokens.c[orig_size], orig_key, has_special,
                              tokens.length - orig_size)

    cdef str _split_affixes(
        self,
        Pool mem,
        str string,
        vector[const LexemeC*] *prefixes,
        vector[const LexemeC*] *suffixes,
        int* has_special,
        bint with_special_cases
    ):
        cdef str prefix
        cdef str suffix
        cdef str minus_pre
        cdef str minus_suf
        cdef size_t last_size = 0
        while string and len(string) != last_size:
            if (rmcp_ascii_word_text(string) if type(self) is Tokenizer and self._rmcp_ascii_word_match and type(string) is str else (self.token_match and self.token_match(string))):
                break
            if with_special_cases and self._specials.get(hash_string(string)) != NULL:
                break
            last_size = len(string)
            pre_len = self.find_prefix(string)
            if pre_len != 0:
                prefix = string[:pre_len]
                minus_pre = string[pre_len:]
                if minus_pre and with_special_cases and self._specials.get(hash_string(minus_pre)) != NULL:
                    string = minus_pre
                    prefixes.push_back(self.vocab.get(mem, prefix))
                    break
            suf_len = self.find_suffix(string[pre_len:])
            if suf_len != 0:
                suffix = string[-suf_len:]
                minus_suf = string[:-suf_len]
                if minus_suf and with_special_cases and self._specials.get(hash_string(minus_suf)) != NULL:
                    string = minus_suf
                    suffixes.push_back(self.vocab.get(mem, suffix))
                    break
            if pre_len and suf_len and (pre_len + suf_len) <= len(string):
                string = string[pre_len:-suf_len]
                prefixes.push_back(self.vocab.get(mem, prefix))
                suffixes.push_back(self.vocab.get(mem, suffix))
            elif pre_len:
                string = minus_pre
                prefixes.push_back(self.vocab.get(mem, prefix))
            elif suf_len:
                string = minus_suf
                suffixes.push_back(self.vocab.get(mem, suffix))
        return string

    cdef int _attach_tokens(self, Doc tokens, str string,
                            vector[const LexemeC*] *prefixes,
                            vector[const LexemeC*] *suffixes,
                            int* has_special,
                            bint with_special_cases) except -1:
        cdef const LexemeC* lexeme
        cdef str span
        cdef int i
        if prefixes.size():
            for i in range(prefixes.size()):
                tokens.push_back(prefixes[0][i], False)
        if string:
            if self._try_specials_and_cache(hash_string(string), tokens, has_special, with_special_cases):
                pass
            elif (
                ((rmcp_ascii_word_text(string) if type(self) is Tokenizer and self._rmcp_ascii_word_match and type(string) is str else (self.token_match and self.token_match(string)))) or
                (self.url_match and self.url_match(string))
            ):

                # We're always saying 'no' to spaces here -- the caller will
                # fix up the outermost one, with reference to the original.
                # See Issue #859
                tokens.push_back(self.vocab.get(tokens.mem, string), False)
            else:
                matches = self.find_infix(string)
                if not matches:
                    tokens.push_back(self.vocab.get(tokens.mem, string), False)
                else:
                    # Let's say we have dyn-o-mite-dave - the regex finds the
                    # start and end positions of the hyphens
                    start = 0
                    start_before_infixes = start
                    for match in matches:
                        infix_start = match.start()
                        infix_end = match.end()

                        if infix_start == start_before_infixes:
                            continue

                        if infix_start != start:
                            span = string[start:infix_start]
                            tokens.push_back(self.vocab.get(tokens.mem, span), False)

                        if infix_start != infix_end:
                            # If infix_start != infix_end, it means the infix
                            # token is non-empty. Empty infix tokens are useful
                            # for tokenization in some languages (see
                            # https://github.com/explosion/spaCy/issues/768)
                            infix_span = string[infix_start:infix_end]
                            tokens.push_back(self.vocab.get(tokens.mem, infix_span), False)
                        start = infix_end
                    span = string[start:]
                    if span:
                        tokens.push_back(self.vocab.get(tokens.mem, span), False)
        cdef vector[const LexemeC*].reverse_iterator it = suffixes.rbegin()
        while it != suffixes.rend():
            lexeme = deref(it)
            preinc(it)
            tokens.push_back(lexeme, False)

    cdef int _save_cached(self, const TokenC* tokens, hash_t key,
                          int* has_special, int n) except -1:
        cdef int i
        if n <= 0:
            # avoid mem alloc of zero length
            return 0
        if self.vocab.in_memory_zone:
            return 0
        # See #1250
        if has_special[0]:
            return 0
        cached = <_Cached*>self.mem.alloc(1, sizeof(_Cached))
        cached.length = n
        cached.is_lex = True
        lexemes = <const LexemeC**>self.mem.alloc(n, sizeof(LexemeC**))
        for i in range(n):
            lexemes[i] = tokens[i].lex
        cached.data.lexemes = <const LexemeC* const*>lexemes
        self._cache.set(key, cached)

    def find_infix(self, str string):
        """Find internal split points of the string, such as hyphens.

        string (str): The string to segment.
        RETURNS (list): A list of `re.MatchObject` objects that have `.start()`
            and `.end()` methods, denoting the placement of internal segment
            separators, e.g. hyphens.

        DOCS: https://spacy.io/api/tokenizer#find_infix
        """
        if self.infix_finditer is None:
            return 0
        return list(self.infix_finditer(string))

    def find_prefix(self, str string):
        """Find the length of a prefix that should be segmented from the
        string, or None if no prefix rules match.

        string (str): The string to segment.
        RETURNS (int): The length of the prefix if present, otherwise `None`.

        DOCS: https://spacy.io/api/tokenizer#find_prefix
        """
        cdef Py_ssize_t native_prefix
        if type(self) is Tokenizer:
            native_prefix = rmcp_known_prefix_length(string, self._prefix_search, _rmcp_prefix_callback_ref)
            if native_prefix >= 0:
                return native_prefix
        if self.prefix_search is None:
            return 0
        match = self.prefix_search(string)
        return (match.end() - match.start()) if match is not None else 0

    def find_suffix(self, str string):
        """Find the length of a suffix that should be segmented from the
        string, or None if no suffix rules match.

        string (str): The string to segment.
        Returns (int): The length of the suffix if present, otherwise `None`.

        DOCS: https://spacy.io/api/tokenizer#find_suffix
        """
        cdef Py_ssize_t native_length
        if (type(self) is Tokenizer and type(string) is str and
            rmcp_suffix_callback(self._suffix_search, _rmcp_suffix_function, RMCPSuffixSearch)):
            # Preserve the original positive branch even after _search deletion.
            if rmcp_single_ascii_suffix(string):
                return 1
            if _rmcp_has_default_suffix_search(self._suffix_search.__self__):
                native_length = rmcp_ascii_english_suffix(string)
                if native_length >= 0:
                    return native_length
        if self.suffix_search is None:
            return 0
        match = self.suffix_search(string)
        return (match.end() - match.start()) if match is not None else 0

    def _load_special_cases(self, special_cases):
        """Add special-case tokenization rules."""
        if special_cases is not None:
            for chunk, substrings in sorted(special_cases.items()):
                self.add_special_case(chunk, substrings)

    def _validate_special_case(self, chunk, substrings):
        """Check whether the `ORTH` fields match the string. Check that
        additional features beyond `ORTH` and `NORM` are not set by the
        exception.

        chunk (str): The string to specially tokenize.
        substrings (iterable): A sequence of dicts, where each dict describes
            a token and its attributes.
        """
        attrs = [intify_attrs(spec, _do_deprecated=True) for spec in substrings]
        orth = "".join([spec[ORTH] for spec in attrs])
        if chunk != orth:
            raise ValueError(Errors.E997.format(chunk=chunk, orth=orth, token_attrs=substrings))
        for substring in attrs:
            for attr in substring:
                if attr not in (ORTH, NORM):
                    raise ValueError(Errors.E1005.format(attr=self.vocab.strings[attr], chunk=chunk))

    def add_special_case(self, str string, substrings):
        """Add a special-case tokenization rule.

        string (str): The string to specially tokenize.
        substrings (iterable): A sequence of dicts, where each dict describes
            a token and its attributes. The `ORTH` fields of the attributes
            must exactly match the string when they are concatenated.

        DOCS: https://spacy.io/api/tokenizer#add_special_case
        """
        self._validate_special_case(string, substrings)
        substrings = list(substrings)
        cached = <_Cached*>self.mem.alloc(1, sizeof(_Cached))
        cached.length = len(substrings)
        cached.is_lex = False
        cached.data.tokens = self.vocab.make_fused_token(substrings)
        key = hash_string(string)
        stale_special = <_Cached*>self._specials.get(key)
        self._specials.set(key, cached)
        if stale_special is not NULL:
            self.mem.free(stale_special)
        self._rules[string] = substrings
        self._flush_cache()
        if not self.faster_heuristics or self.find_prefix(string) or self.find_infix(string) or self.find_suffix(string) or " " in string:
            self._special_matcher.add(string, None, self._tokenize_affixes(string, False))

    def _reload_special_cases(self):
        self._flush_cache()
        self._flush_specials()
        self._load_special_cases(self._rules)

    def explain(self, text):
        """A debugging tokenizer that provides information about which
        tokenizer rule or pattern was matched for each token. The tokens
        produced are identical to `nlp.tokenizer()` except for whitespace
        tokens.

        string (str): The string to tokenize.
        RETURNS (list): A list of (pattern_string, token_string) tuples

        DOCS: https://spacy.io/api/tokenizer#explain
        """
        prefix_search = self.prefix_search
        if prefix_search is None:
            prefix_search = re.compile("a^").search
        suffix_search = self.suffix_search
        if suffix_search is None:
            suffix_search = re.compile("a^").search
        infix_finditer = self.infix_finditer
        if infix_finditer is None:
            infix_finditer = re.compile("a^").finditer
        token_match = self.token_match
        if token_match is None:
            token_match = re.compile("a^").match
        url_match = self.url_match
        if url_match is None:
            url_match = re.compile("a^").match
        special_cases = {}
        for orth, special_tokens in self.rules.items():
            special_cases[orth] = [intify_attrs(special_token, strings_map=self.vocab.strings, _do_deprecated=True) for special_token in special_tokens]
        tokens = []
        for substring in text.split():
            suffixes = []
            while substring:
                if substring in special_cases:
                    tokens.extend(("SPECIAL-" + str(i + 1), self.vocab.strings[e[ORTH]]) for i, e in enumerate(special_cases[substring]))
                    substring = ''
                    continue
                while prefix_search(substring) or suffix_search(substring):
                    if token_match(substring):
                        tokens.append(("TOKEN_MATCH", substring))
                        substring = ''
                        break
                    if substring in special_cases:
                        tokens.extend(("SPECIAL-" + str(i + 1), self.vocab.strings[e[ORTH]]) for i, e in enumerate(special_cases[substring]))
                        substring = ''
                        break
                    if prefix_search(substring):
                        split = prefix_search(substring).end()
                        # break if pattern matches the empty string
                        if split == 0:
                            break
                        tokens.append(("PREFIX", substring[:split]))
                        substring = substring[split:]
                        if substring in special_cases:
                            continue
                    if suffix_search(substring):
                        split = suffix_search(substring).start()
                        # break if pattern matches the empty string
                        if split == len(substring):
                            break
                        suffixes.append(("SUFFIX", substring[split:]))
                        substring = substring[:split]
                if len(substring) == 0:
                    continue
                if token_match(substring):
                    tokens.append(("TOKEN_MATCH", substring))
                    substring = ''
                elif url_match(substring):
                    tokens.append(("URL_MATCH", substring))
                    substring = ''
                elif substring in special_cases:
                    tokens.extend((f"SPECIAL-{i + 1}", self.vocab.strings[e[ORTH]]) for i, e in enumerate(special_cases[substring]))
                    substring = ''
                elif list(infix_finditer(substring)):
                    infixes = infix_finditer(substring)
                    offset = 0
                    for match in infixes:
                        if offset == 0 and match.start() == 0:
                            continue
                        if substring[offset : match.start()]:
                            tokens.append(("TOKEN", substring[offset : match.start()]))
                        if substring[match.start() : match.end()]:
                            tokens.append(("INFIX", substring[match.start() : match.end()]))
                        offset = match.end()
                    if substring[offset:]:
                        tokens.append(("TOKEN", substring[offset:]))
                    substring = ''
                elif substring:
                    tokens.append(("TOKEN", substring))
                    substring = ''
            tokens.extend(reversed(suffixes))
        # Find matches for special cases handled by special matcher
        words, spaces = get_words_and_spaces([t[1] for t in tokens], text)
        t_words = []
        t_spaces = []
        for word, space in zip(words, spaces):
            if not word.isspace():
                t_words.append(word)
                t_spaces.append(space)
        doc = Doc(self.vocab, words=t_words, spaces=t_spaces)
        matches = self._special_matcher(doc)
        spans = [Span(doc, s, e, label=m_id) for m_id, s, e in matches]
        spans = util.filter_spans(spans)
        # Replace matched tokens with their exceptions
        i = 0
        final_tokens = []
        spans_by_start = {s.start: s for s in spans}
        while i < len(tokens):
            if i in spans_by_start:
                span = spans_by_start[i]
                exc = [d[ORTH] for d in special_cases[span.label_]]
                # The phrase matcher can overmatch for tokens separated by
                # spaces in the text but not in the underlying rule, so skip
                # cases where the texts aren't identical
                if span.text != "".join([self.vocab.strings[orth] for orth in exc]):
                    final_tokens.append(tokens[i])
                    i += 1
                else:
                    for j, orth in enumerate(exc):
                        final_tokens.append((f"SPECIAL-{j + 1}", self.vocab.strings[orth]))
                    i += len(span)
            else:
                final_tokens.append(tokens[i])
                i += 1
        return final_tokens

    def score(self, examples, **kwargs):
        validate_examples(examples, "Tokenizer.score")
        return Scorer.score_tokenization(examples)

    def to_disk(self, path, **kwargs):
        """Save the current state to a directory.

        path (str / Path): A path to a directory, which will be created if
            it doesn't exist.
        exclude (list): String names of serialization fields to exclude.

        DOCS: https://spacy.io/api/tokenizer#to_disk
        """
        path = util.ensure_path(path)
        with path.open("wb") as file_:
            file_.write(self.to_bytes(**kwargs))

    def from_disk(self, path, *, exclude=tuple()):
        """Loads state from a directory. Modifies the object in place and
        returns it.

        path (str / Path): A path to a directory.
        exclude (list): String names of serialization fields to exclude.
        RETURNS (Tokenizer): The modified `Tokenizer` object.

        DOCS: https://spacy.io/api/tokenizer#from_disk
        """
        path = util.ensure_path(path)
        with path.open("rb") as file_:
            bytes_data = file_.read()
        self.from_bytes(bytes_data, exclude=exclude)
        return self

    def to_bytes(self, *, exclude=tuple()):
        """Serialize the current state to a binary string.

        exclude (list): String names of serialization fields to exclude.
        RETURNS (bytes): The serialized form of the `Tokenizer` object.

        DOCS: https://spacy.io/api/tokenizer#to_bytes
        """
        serializers = {
            "vocab": lambda: self.vocab.to_bytes(exclude=exclude),
            "prefix_search": lambda: _get_regex_pattern(self.prefix_search),
            "suffix_search": lambda: _get_regex_pattern(self.suffix_search),
            "infix_finditer": lambda: _get_regex_pattern(self.infix_finditer),
            "token_match": lambda: _get_regex_pattern(self.token_match),
            "url_match": lambda: _get_regex_pattern(self.url_match),
            "exceptions": lambda: dict(sorted(self._rules.items())),
            "faster_heuristics": lambda: self.faster_heuristics,
        }
        return util.to_bytes(serializers, exclude)

    def from_bytes(self, bytes_data, *, exclude=tuple()):
        """Load state from a binary string.

        bytes_data (bytes): The data to load from.
        exclude (list): String names of serialization fields to exclude.
        RETURNS (Tokenizer): The `Tokenizer` object.

        DOCS: https://spacy.io/api/tokenizer#from_bytes
        """
        data = {}
        deserializers = {
            "vocab": lambda b: self.vocab.from_bytes(b, exclude=exclude),
            "prefix_search": lambda b: data.setdefault("prefix_search", b),
            "suffix_search": lambda b: data.setdefault("suffix_search", b),
            "infix_finditer": lambda b: data.setdefault("infix_finditer", b),
            "token_match": lambda b: data.setdefault("token_match", b),
            "url_match": lambda b: data.setdefault("url_match", b),
            "exceptions": lambda b: data.setdefault("rules", b),
            "faster_heuristics": lambda b: data.setdefault("faster_heuristics", b),
        }
        # reset all properties and flush all caches (through rules),
        # reset rules first so that _reload_special_cases is trivial/fast as
        # the other properties are reset
        self.rules = {}
        self.prefix_search = None
        self.suffix_search = None
        self.infix_finditer = None
        self.token_match = None
        self.url_match = None
        util.from_bytes(bytes_data, deserializers, exclude)
        if "prefix_search" in data and isinstance(data["prefix_search"], str):
            self.prefix_search = re.compile(data["prefix_search"]).search
        if "suffix_search" in data and isinstance(data["suffix_search"], str):
            self.suffix_search = re.compile(data["suffix_search"]).search
        if "infix_finditer" in data and isinstance(data["infix_finditer"], str):
            self.infix_finditer = re.compile(data["infix_finditer"]).finditer
        if "token_match" in data and isinstance(data["token_match"], str):
            self.token_match = re.compile(data["token_match"]).match
        if "url_match" in data and isinstance(data["url_match"], str):
            self.url_match = re.compile(data["url_match"]).match
        if "faster_heuristics" in data:
            self.faster_heuristics = data["faster_heuristics"]
        # always load rules last so that all other settings are set before the
        # internal tokenization for the phrase matcher
        if "rules" in data and isinstance(data["rules"], dict):
            self.rules = data["rules"]
        return self


def _get_regex_pattern(regex):
    """Get a pattern string for a regex, or None if the pattern is None."""
    return None if regex is None else regex.__self__.pattern


cdef extern from "<algorithm>" namespace "std" nogil:
    void stdsort "sort"(vector[SpanC].iterator,
                        vector[SpanC].iterator,
                        bint (*)(SpanC, SpanC))


cdef bint len_start_cmp(SpanC a, SpanC b) noexcept nogil:
    if a.end - a.start == b.end - b.start:
        return b.start < a.start
    return a.end - a.start < b.end - b.start


cdef bint start_cmp(SpanC a, SpanC b) noexcept nogil:
    return a.start < b.start
