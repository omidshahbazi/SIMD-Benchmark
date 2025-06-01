#include <inttypes.h>
#include <type_traits>
#include <immintrin.h>
#include <initializer_list>
#include <iostream>
#include <array>
#include <chrono>

template<typename Type>
concept MatrixValueType = std::is_floating_point_v<Type>;

template<uint8_t NumType>
concept MatrixDimensionType = (2 <= NumType && NumType <= 4);

template<MatrixValueType ValueType, uint8_t Dimension, bool UseSIMD> requires MatrixDimensionType<Dimension>
struct SquaredMatrix
{
public:
	SquaredMatrix() :
		Data{}
	{
	}

	SquaredMatrix(std::initializer_list<ValueType> List) :
		Data{}
	{
		uint8_t i = 0;
		for (ValueType value : List)
		{
			Data[i++] = value;
		}
	}

	inline SquaredMatrix operator*(const SquaredMatrix& Other)
	{
		SquaredMatrix result;

		if constexpr (UseSIMD)
		{
			static_assert(sizeof(ValueType) == 4);
			static_assert(Dimension == 4);

			for (uint8_t row = 0; row < Dimension; ++row)
			{
				__m128 rowData = _mm_load_ps(Cells[row]);

				for (uint8_t column = 0; column < Dimension; ++column)
				{
					std::array<ValueType, Dimension> arr;
					for (uint8_t x = 0; x < Dimension; ++x)
						arr[x] = Other.Cells[x][column];

					__m128 columnData = _mm_load_ps(arr.data());

					__m128 resultData = _mm_dp_ps(rowData, columnData, 0xF1);

					result.Cells[row][column] += _mm_cvtss_f32(resultData);
				}
			}
		}
		else
		{
			for (uint8_t row = 0; row < Dimension; ++row)
				for (uint8_t column = 0; column < Dimension; ++column)
				{
					for (uint8_t x = 0; x < Dimension; ++x)
						result.Cells[row][column] += Cells[row][x] * Other.Cells[x][column];
				}
		}

		return result;
	}

public:
	union
	{
		ValueType Cells[Dimension][Dimension];
		ValueType Data[Dimension * Dimension];
	};

	static const SquaredMatrix Identity;
};

template<MatrixValueType ValueType, uint8_t Dimension, bool UseSIMD> requires MatrixDimensionType<Dimension>
const SquaredMatrix<ValueType, Dimension, UseSIMD> SquaredMatrix<ValueType, Dimension, UseSIMD>::Identity = []
	{
		SquaredMatrix<ValueType, Dimension, UseSIMD> result;

		for (uint8_t row = 0; row < Dimension; ++row)
			for (uint8_t column = 0; column < Dimension; ++column)
				result.Cells[row][column] = (row == column);

		return result;
	}();

template<MatrixValueType ValueType, uint8_t Dimension, bool UseSIMD>
void Print(const SquaredMatrix<ValueType, Dimension, UseSIMD>& Matrix)
{
	for (uint8_t row = 0; row < Dimension; ++row)
	{
		for (uint8_t column = 0; column < Dimension; ++column)
			printf("%f, ", Matrix.Cells[row][column]);

		printf("\n");
	}

	printf("\n");
}

typedef std::chrono::duration<long long, std::pico> picosecond;

template<bool UseSIMD>
picosecond Benchmark()
{
	typedef SquaredMatrix<float, 4, UseSIMD> MatrixF4x4;

	MatrixF4x4 mat1 = MatrixF4x4::Identity;
	MatrixF4x4 mat2 = { 1, 2, 3, 4, 5,6,7, 8, 9, 10, 11, 12,13, 14, 15,16 };

	const uint32_t NumSteps = 1'000'000'000;

	std::chrono::time_point start = std::chrono::high_resolution_clock::now();

	MatrixF4x4 total;
	for (uint32_t i = 0; i < NumSteps; ++i)
	{
		total = MatrixF4x4::Identity;
	}

	picosecond ctorElapsed = std::chrono::high_resolution_clock::now() - start;

	start = std::chrono::high_resolution_clock::now();

	for (uint32_t i = 0; i < NumSteps; ++i)
	{
		total = mat1 * mat2;
	}

	picosecond elapsed = (std::chrono::high_resolution_clock::now() - start) - ctorElapsed;

	if constexpr (UseSIMD)
		printf("SIMD\n", elapsed.count());
	else
		printf("Simple\n", elapsed.count());

	printf("_________________________________________________________________\n");

	Print(mat1 * mat2);

	printf("Total Time: ~%lldns Per-Multiplication Time: %ipsec\n", std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count(), (elapsed / NumSteps).count());

	printf("-----------------------------------------------------------------\n");

	return elapsed;
}

int main()
{
	picosecond timeSimple = Benchmark<false>();
	picosecond timeSIMD = Benchmark<true>();

	printf("SIMD version is %f%% faster than Simple version\n", ((float)timeSimple.count() / timeSIMD.count() - 1) * 100);
}