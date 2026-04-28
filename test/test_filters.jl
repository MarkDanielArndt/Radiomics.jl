@testset "haar_wavelet" begin

    # ---- 2D -----------------------------------------------------------------

    @testset "haar_wavelet_2d – all subbands" begin
        img = Float32.(reshape(1:4096, 64, 64))
        sb = Radiomics.haar_wavelet_2d(img)
        @test Set(keys(sb)) == Set(["LL", "LH", "HL", "HH"])
        for (_, v) in sb
            @test size(v) == size(img)
        end
        # constant image → all high-pass subbands must be zero
        flat = ones(Float32, 64, 64)
        sb_flat = Radiomics.haar_wavelet_2d(flat)
        @test all(sb_flat["LH"] .≈ 0)
        @test all(sb_flat["HL"] .≈ 0)
        @test all(sb_flat["HH"] .≈ 0)
    end

    @testset "haar_wavelet_2d – single subband" begin
        img = rand(Float32, 64, 64)
        for name in ("LL", "LH", "HL", "HH")
            result = Radiomics.haar_wavelet_2d(img; subband=name)
            @test result isa Matrix{Float64}
            @test size(result) == size(img)
            # must match the corresponding entry from the full decomposition
            @test result ≈ Radiomics.haar_wavelet_2d(img)[name]
        end
        @test_throws ErrorException Radiomics.haar_wavelet_2d(img; subband="LLL")
        @test_throws ErrorException Radiomics.haar_wavelet_2d(img; subband="XY")
    end

    # ---- 3D -----------------------------------------------------------------

    @testset "haar_wavelet_3d – all subbands" begin
        img = rand(Float32, 64, 64, 64)
        sb = Radiomics.haar_wavelet_3d(img)
        expected = Set(["LLL","LLH","LHL","LHH","HLL","HLH","HHL","HHH"])
        @test Set(keys(sb)) == expected
        for (_, v) in sb
            @test size(v) == size(img)
        end
        flat = ones(Float32, 64, 64, 64)
        sb_flat = Radiomics.haar_wavelet_3d(flat)
        for name in ("LLH","LHL","LHH","HLL","HLH","HHL","HHH")
            @test all(sb_flat[name] .≈ 0)
        end
    end

    @testset "haar_wavelet_3d – single subband" begin
        img = rand(Float32, 64, 64, 64)
        for name in ("LLL", "LLH", "HHL", "HHH")
            result = Radiomics.haar_wavelet_3d(img; subband=name)
            @test result isa Array{Float64, 3}
            @test size(result) == size(img)
            @test result ≈ Radiomics.haar_wavelet_3d(img)[name]
        end
        @test_throws ErrorException Radiomics.haar_wavelet_3d(img; subband="LL")
        @test_throws ErrorException Radiomics.haar_wavelet_3d(img; subband="XYZ")
    end

    # ---- dispatch ------------------------------------------------------------

    @testset "haar_wavelet dispatch" begin
        img2 = rand(Float32, 64, 64)
        img3 = rand(Float32, 64, 64, 64)
        @test Radiomics.haar_wavelet(img2) isa Dict
        @test Radiomics.haar_wavelet(img3) isa Dict
        @test Radiomics.haar_wavelet(img2; subband="LH") isa Matrix
        @test Radiomics.haar_wavelet(img3; subband="LLH") isa Array{Float64, 3}
        @test_throws ErrorException Radiomics.haar_wavelet(rand(Float32, 64, 64, 64, 64))
    end

end
